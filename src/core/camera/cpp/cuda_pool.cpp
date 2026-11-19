// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera/cuda_pool.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>

#ifdef CAMERA_HAS_CUDA
#    include <cstring>
#    include <cuda.h>
#    include <dlfcn.h>
#endif

namespace core::camera
{

namespace
{

void check_shape(std::uint32_t slot_count, std::uint64_t slot_bytes)
{
    if (slot_count == 0 || slot_count > MAX_SLOTS_PER_STREAM || slot_bytes == 0 || slot_bytes > MAX_SLOT_BYTES)
    {
        throw std::invalid_argument("a CUDA pool needs between 1 and " + std::to_string(MAX_SLOTS_PER_STREAM) +
                                    " slots of between 1 and " + std::to_string(MAX_SLOT_BYTES) + " bytes");
    }
}

#ifdef CAMERA_HAS_CUDA

// Slots start a multiple of this apart, for both sides of a pool.
constexpr std::uint64_t SLOT_ALIGNMENT = 4096;

std::uint64_t stride_of(std::uint64_t slot_bytes)
{
    return (slot_bytes + SLOT_ALIGNMENT - 1) / SLOT_ALIGNMENT * SLOT_ALIGNMENT;
}

// The driver's functions this file calls. They are looked up in the driver's library when CUDA
// memory is first used, so a process that never uses it runs where no NVIDIA driver is
// installed. The names are the header's, which maps some of them to a versioned symbol.
#    define CAMERA_CUDA_FUNCTIONS(X)                                                                                   \
        X(cuInit)                                                                                                      \
        X(cuGetErrorName)                                                                                              \
        X(cuDeviceGetCount)                                                                                            \
        X(cuDeviceGet)                                                                                                 \
        X(cuDeviceGetUuid)                                                                                             \
        X(cuDevicePrimaryCtxRetain)                                                                                    \
        X(cuDevicePrimaryCtxRelease)                                                                                   \
        X(cuCtxPushCurrent)                                                                                            \
        X(cuCtxPopCurrent)                                                                                             \
        X(cuCtxSynchronize)                                                                                            \
        X(cuMemGetAllocationGranularity)                                                                               \
        X(cuMemCreate)                                                                                                 \
        X(cuMemRelease)                                                                                                \
        X(cuMemAddressReserve)                                                                                         \
        X(cuMemAddressFree)                                                                                            \
        X(cuMemMap)                                                                                                    \
        X(cuMemUnmap)                                                                                                  \
        X(cuMemSetAccess)                                                                                              \
        X(cuMemExportToShareableHandle)                                                                                \
        X(cuMemImportFromShareableHandle)                                                                              \
        X(cuMemcpyHtoD)                                                                                                \
        X(cuMemcpyDtoH)

#    define CAMERA_CUDA_TEXT_OF(name) #    name
#    define CAMERA_CUDA_SYMBOL(name) CAMERA_CUDA_TEXT_OF(name)

struct Driver
{
#    define X(name) decltype(&::name) name = nullptr;
    CAMERA_CUDA_FUNCTIONS(X)
#    undef X

    std::string unavailable; // why CUDA cannot be used; empty when it can
    int device_count = 0;
};

Driver load_driver()
{
    Driver driver;
    // Never closed: the driver keeps state for the life of the process.
    void* const library = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        driver.unavailable = "the NVIDIA driver's CUDA library, libcuda.so.1, cannot be loaded";
        return driver;
    }
#    define X(name)                                                                                                    \
        driver.name = reinterpret_cast<decltype(driver.name)>(::dlsym(library, CAMERA_CUDA_SYMBOL(name)));             \
        if (driver.name == nullptr && driver.unavailable.empty())                                                      \
        {                                                                                                              \
            driver.unavailable = "the NVIDIA driver is too old for shared CUDA memory: libcuda.so.1 has no " +         \
                                 std::string(CAMERA_CUDA_SYMBOL(name));                                                \
        }
    CAMERA_CUDA_FUNCTIONS(X)
#    undef X
    if (!driver.unavailable.empty())
    {
        return driver;
    }
    const CUresult started = driver.cuInit(0);
    if (started != CUDA_SUCCESS || driver.cuDeviceGetCount(&driver.device_count) != CUDA_SUCCESS ||
        driver.device_count <= 0)
    {
        const char* name = nullptr;
        driver.cuGetErrorName(started, &name);
        driver.unavailable = started != CUDA_SUCCESS ? "the NVIDIA driver cannot start CUDA (" +
                                                           std::string(name != nullptr ? name : "?") + ")" :
                                                       "the NVIDIA driver finds no CUDA device";
    }
    return driver;
}

const Driver& driver()
{
    static const Driver loaded = load_driver();
    return loaded;
}

const Driver& usable_driver()
{
    const Driver& loaded = driver();
    if (!loaded.unavailable.empty())
    {
        throw std::runtime_error("CUDA memory cannot be used: " + loaded.unavailable);
    }
    return loaded;
}

void must(CUresult result, const char* what)
{
    if (result != CUDA_SUCCESS)
    {
        const char* name = nullptr;
        driver().cuGetErrorName(result, &name);
        throw std::runtime_error(std::string("CUDA: ") + what + " failed (" + (name != nullptr ? name : "?") + ")");
    }
}

struct Device
{
    CUdevice handle = 0;
    std::uint32_t index = 0;
};

Device device_of(const CudaDeviceUuid& uuid)
{
    const Driver& cuda = usable_driver();
    for (int index = 0; index < cuda.device_count; ++index)
    {
        Device device{ 0, static_cast<std::uint32_t>(index) };
        CUuuid found = {};
        if (cuda.cuDeviceGet(&device.handle, index) == CUDA_SUCCESS &&
            cuda.cuDeviceGetUuid(&found, device.handle) == CUDA_SUCCESS &&
            std::memcmp(found.bytes, uuid.data(), uuid.size()) == 0)
        {
            return device;
        }
    }
    throw std::runtime_error("CUDA device " + cuda_device_name(uuid) +
                             " is not one of this process's devices; a camera's CUDA memory is used on the "
                             "device it is on");
}

// A device's primary context, kept for as long as the object exists.
class PrimaryContext
{
public:
    explicit PrimaryContext(CUdevice device) : m_device(device)
    {
        must(driver().cuDevicePrimaryCtxRetain(&m_context, device), "taking the device's primary context");
    }

    ~PrimaryContext()
    {
        driver().cuDevicePrimaryCtxRelease(m_device);
    }

    PrimaryContext(const PrimaryContext&) = delete;
    PrimaryContext& operator=(const PrimaryContext&) = delete;

    CUcontext get() const
    {
        return m_context;
    }

private:
    CUdevice m_device;
    CUcontext m_context = nullptr;
};

// Makes a context the calling thread's for a scope, and puts back what the thread had.
class CurrentContext
{
public:
    explicit CurrentContext(CUcontext context)
    {
        must(driver().cuCtxPushCurrent(context), "making the device's context current");
    }

    ~CurrentContext()
    {
        CUcontext previous = nullptr;
        driver().cuCtxPopCurrent(&previous);
    }

    CurrentContext(const CurrentContext&) = delete;
    CurrentContext& operator=(const CurrentContext&) = delete;
};

CUmemLocation location_of(const Device& device)
{
    CUmemLocation location = {};
    location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    location.id = static_cast<int>(device.index);
    return location;
}

CUmemAllocationProp shareable_memory_on(const Device& device)
{
    CUmemAllocationProp properties = {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location = location_of(device);
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    return properties;
}

std::uint64_t allocation_bytes_on(const Device& device, std::uint32_t slot_count, std::uint64_t slot_bytes)
{
    const CUmemAllocationProp properties = shareable_memory_on(device);
    std::size_t granularity = 0;
    must(driver().cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
         "asking for the device's allocation granularity");
    const std::uint64_t wanted = stride_of(slot_bytes) * slot_count;
    return (wanted + granularity - 1) / granularity * granularity;
}

// One allocation mapped into this process: the range of addresses and what backs it. Both
// kinds of pool hold one; they differ in where the allocation comes from and in the access.
struct Mapping
{
    explicit Mapping(const Device& on) : device(on), context(on.handle)
    {
    }

    ~Mapping()
    {
        // The context may be gone with the process's CUDA state at exit; nothing to do then.
        CUcontext previous = nullptr;
        if (driver().cuCtxPushCurrent(context.get()) != CUDA_SUCCESS)
        {
            return;
        }
        if (mapped)
        {
            driver().cuMemUnmap(base, bytes);
        }
        if (base != 0)
        {
            driver().cuMemAddressFree(base, bytes);
        }
        if (allocation != 0)
        {
            driver().cuMemRelease(allocation);
        }
        driver().cuCtxPopCurrent(&previous);
    }

    // Reserves addresses for the allocation and maps it with the given access.
    void map(CUmemAccess_flags access)
    {
        must(driver().cuMemAddressReserve(&base, bytes, 0, 0, 0), "reserving addresses for a pool");
        must(driver().cuMemMap(base, bytes, 0, allocation, 0), "mapping a pool");
        mapped = true;
        CUmemAccessDesc description = {};
        description.location = location_of(device);
        description.flags = access;
        must(driver().cuMemSetAccess(base, bytes, &description, 1), "giving access to a pool");
    }

    CudaAddress slot(std::uint32_t index) const
    {
        if (index >= slot_count)
        {
            throw std::out_of_range("slot " + std::to_string(index) + " of a CUDA pool with " +
                                    std::to_string(slot_count) + " slots");
        }
        return static_cast<CudaAddress>(base) + index * stride_of(slot_bytes);
    }

    Device device;
    PrimaryContext context;
    CUmemGenericAllocationHandle allocation = 0;
    CUdeviceptr base = 0;
    bool mapped = false;
    std::uint64_t bytes = 0;
    std::uint32_t slot_count = 0;
    std::uint64_t slot_bytes = 0;
};

#else

const char* const NOT_BUILT =
    "this build of the camera library has no CUDA support (the CUDA Toolkit was not "
    "found when it was built)";

[[noreturn]] void not_built()
{
    throw std::runtime_error(std::string("CUDA memory cannot be used: ") + NOT_BUILT);
}

#endif

} // namespace

std::string cuda_device_name(const CudaDeviceUuid& device)
{
    // The form nvidia-smi prints.
    char text[48];
    std::snprintf(text, sizeof(text), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  device[0], device[1], device[2], device[3], device[4], device[5], device[6], device[7], device[8],
                  device[9], device[10], device[11], device[12], device[13], device[14], device[15]);
    return text;
}

#ifdef CAMERA_HAS_CUDA

std::string cuda_unavailable()
{
    return driver().unavailable;
}

std::uint32_t cuda_device_count()
{
    return static_cast<std::uint32_t>(usable_driver().device_count);
}

CudaDeviceUuid cuda_device_uuid(std::uint32_t index)
{
    const Driver& cuda = usable_driver();
    if (index >= static_cast<std::uint32_t>(cuda.device_count))
    {
        throw std::runtime_error("there is no CUDA device " + std::to_string(index) + "; this process sees " +
                                 std::to_string(cuda.device_count));
    }
    CUdevice device = 0;
    CUuuid found = {};
    must(cuda.cuDeviceGet(&device, static_cast<int>(index)), "opening a device");
    must(cuda.cuDeviceGetUuid(&found, device), "asking a device for its UUID");
    CudaDeviceUuid uuid;
    std::memcpy(uuid.data(), found.bytes, uuid.size());
    return uuid;
}

void copy_to_cuda(const CudaDeviceUuid& device, CudaAddress to, std::span<const std::byte> from)
{
    const PrimaryContext context(device_of(device).handle);
    const CurrentContext current(context.get());
    must(driver().cuMemcpyHtoD(static_cast<CUdeviceptr>(to), from.data(), from.size()), "copying to a device");
    // The copy returns once the bytes are on their way; another process sees them when the
    // device has finished, and that is what a camera says by publishing.
    must(driver().cuCtxSynchronize(), "waiting for a copy to a device");
}

void copy_from_cuda(const CudaDeviceUuid& device, std::span<std::byte> to, CudaAddress from)
{
    const PrimaryContext context(device_of(device).handle);
    const CurrentContext current(context.get());
    must(driver().cuMemcpyDtoH(to.data(), static_cast<CUdeviceptr>(from), to.size()), "copying from a device");
}

struct CudaPool::Impl
{
    explicit Impl(const Device& device) : mapping(device)
    {
    }

    Mapping mapping;
    UniqueFd descriptor;
};

CudaPool::CudaPool(const CudaDeviceUuid& device, std::uint32_t slot_count, std::uint64_t slot_bytes)
{
    check_shape(slot_count, slot_bytes);
    m_impl = std::make_unique<Impl>(device_of(device));
    Mapping& mapping = m_impl->mapping;
    mapping.slot_count = slot_count;
    mapping.slot_bytes = slot_bytes;

    const CurrentContext current(mapping.context.get());
    mapping.bytes = allocation_bytes_on(mapping.device, slot_count, slot_bytes);
    const CUmemAllocationProp properties = shareable_memory_on(mapping.device);
    must(driver().cuMemCreate(&mapping.allocation, mapping.bytes, &properties, 0), "allocating a pool");
    mapping.map(CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    int descriptor = -1;
    must(driver().cuMemExportToShareableHandle(
             &descriptor, mapping.allocation, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
         "exporting a pool");
    m_impl->descriptor.reset(descriptor);
}

CudaPool::~CudaPool() = default;

std::uint32_t CudaPool::slot_count() const
{
    return m_impl->mapping.slot_count;
}

std::uint64_t CudaPool::slot_bytes() const
{
    return m_impl->mapping.slot_bytes;
}

CudaAddress CudaPool::slot(std::uint32_t index) const
{
    return m_impl->mapping.slot(index);
}

int CudaPool::descriptor() const
{
    return m_impl->descriptor.get();
}

struct CudaPoolView::Impl
{
    explicit Impl(const Device& device) : mapping(device)
    {
    }

    Mapping mapping;
};

CudaPoolView::CudaPoolView(UniqueFd descriptor,
                           const CudaDeviceUuid& device,
                           std::uint32_t slot_count,
                           std::uint64_t slot_bytes)
{
    check_shape(slot_count, slot_bytes);
    m_impl = std::make_unique<Impl>(device_of(device));
    Mapping& mapping = m_impl->mapping;
    mapping.slot_count = slot_count;
    mapping.slot_bytes = slot_bytes;

    const CurrentContext current(mapping.context.get());
    mapping.bytes = allocation_bytes_on(mapping.device, slot_count, slot_bytes);
    must(driver().cuMemImportFromShareableHandle(&mapping.allocation,
                                                 reinterpret_cast<void*>(static_cast<std::uintptr_t>(descriptor.get())),
                                                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
         "importing a pool");
    mapping.map(CU_MEM_ACCESS_FLAGS_PROT_READ);
    // The mapping holds the memory from here on; `descriptor` closes as it goes out of scope.
}

CudaPoolView::~CudaPoolView() = default;
CudaPoolView::CudaPoolView(CudaPoolView&& other) noexcept = default;
CudaPoolView& CudaPoolView::operator=(CudaPoolView&& other) noexcept = default;

std::uint32_t CudaPoolView::slot_count() const
{
    return m_impl->mapping.slot_count;
}

std::uint32_t CudaPoolView::device_index() const
{
    return m_impl->mapping.device.index;
}

CudaAddress CudaPoolView::slot(std::uint32_t index) const
{
    return m_impl->mapping.slot(index);
}

#else

std::string cuda_unavailable()
{
    return NOT_BUILT;
}

std::uint32_t cuda_device_count()
{
    not_built();
}

CudaDeviceUuid cuda_device_uuid(std::uint32_t)
{
    not_built();
}

void copy_to_cuda(const CudaDeviceUuid&, CudaAddress, std::span<const std::byte>)
{
    not_built();
}

void copy_from_cuda(const CudaDeviceUuid&, std::span<std::byte>, CudaAddress)
{
    not_built();
}

struct CudaPool::Impl
{
};

CudaPool::CudaPool(const CudaDeviceUuid&, std::uint32_t slot_count, std::uint64_t slot_bytes)
{
    check_shape(slot_count, slot_bytes);
    not_built();
}

CudaPool::~CudaPool() = default;

std::uint32_t CudaPool::slot_count() const
{
    return 0;
}

std::uint64_t CudaPool::slot_bytes() const
{
    return 0;
}

CudaAddress CudaPool::slot(std::uint32_t) const
{
    not_built();
}

int CudaPool::descriptor() const
{
    return -1;
}

struct CudaPoolView::Impl
{
};

CudaPoolView::CudaPoolView(UniqueFd, const CudaDeviceUuid&, std::uint32_t slot_count, std::uint64_t slot_bytes)
{
    check_shape(slot_count, slot_bytes);
    not_built();
}

CudaPoolView::~CudaPoolView() = default;
CudaPoolView::CudaPoolView(CudaPoolView&& other) noexcept = default;
CudaPoolView& CudaPoolView::operator=(CudaPoolView&& other) noexcept = default;

std::uint32_t CudaPoolView::slot_count() const
{
    return 0;
}

std::uint32_t CudaPoolView::device_index() const
{
    return 0;
}

CudaAddress CudaPoolView::slot(std::uint32_t) const
{
    not_built();
}

#endif

} // namespace core::camera
