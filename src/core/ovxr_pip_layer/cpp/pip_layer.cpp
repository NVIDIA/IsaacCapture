// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Application-owned camera quads on a borrowed OVXR session/device. The frame
// owner uploads and returns layer descriptors before OVXR submits. This helper
// never creates a session or intercepts OpenXR frame calls.
#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>

// Keep Vulkan declarations before OpenXR platform structs, including after formatting.
#include <log_bridge/logger.hpp>
#include <openxr/openxr_platform.h>

#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
#    include <cuda_runtime_api.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string_view>
#include <vector>

namespace
{
constexpr size_t kMaxCameras = 3;

// Integer C ABI mirrored by _UploadResult in openxr_quad.py.
enum UploadResult : int
{
    Success = 0,
    InvalidArgument = -1,
    NotAttached = -2,
    CameraUnavailable = -3,
    SwapchainFailed = -4,
    AllocationFailed = -5,
    UploadFailed = -6,
    CompletionUnproved = -7,
    CudaUnavailable = -8,
    UploadModeMismatch = -9,
};

struct Dispatch
{
    PFN_xrGetInstanceProcAddr gipa = nullptr;
    PFN_xrCreateReferenceSpace createSpace = nullptr;
    PFN_xrDestroySpace destroySpace = nullptr;
    PFN_xrEnumerateSwapchainFormats formats = nullptr;
    PFN_xrCreateSwapchain createSwapchain = nullptr;
    PFN_xrDestroySwapchain destroySwapchain = nullptr;
    PFN_xrEnumerateSwapchainImages images = nullptr;
    PFN_xrAcquireSwapchainImage acquire = nullptr;
    PFN_xrWaitSwapchainImage wait = nullptr;
    PFN_xrReleaseSwapchainImage release = nullptr;
};

struct CameraSession
{
    XrInstance instance = XR_NULL_HANDLE;
    XrSession session = XR_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    XrSpace view = XR_NULL_HANDLE;
    XrSpace local = XR_NULL_HANDLE;
    XrSwapchain swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageVulkanKHR> images;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* pixels = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t acquiredIndex = 0;
    bool acquired = false;
    bool waited = false;
    bool fencePending = false;
    bool retired = false;
    bool ready = false;
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    cudaExternalMemory_t cudaMemory = nullptr;
    void* cudaPixels = nullptr;
    cudaStream_t cudaStream = nullptr;
    int cudaDevice = -1;
    int vulkanCudaDevice = -1;
    bool cudaPending = false;
#endif
};

struct Panel
{
    bool enabled = true;
    bool headLocked = true;
    XrPosef pose{ { 0, 0, 0, 1 }, { 0, -0.15f, -0.8f } };
    XrExtent2Df size{ 0.48f, 0.30f };
};

// Each frame consumer owns one helper; calls stay on its XR owner thread.
struct PanelSession
{
    Dispatch dispatch{};
    std::array<CameraSession, kMaxCameras> cameras{};
    std::array<Panel, kMaxCameras> panels{};
    std::array<XrCompositionLayerQuad, kMaxCameras> quads{};
    bool attached = false;
};
constexpr uint32_t kMaxImageDimension = 4096;

template <typename Function>
bool resolve(PFN_xrGetInstanceProcAddr gipa, XrInstance instance, const char* name, Function& target)
{
    PFN_xrVoidFunction raw = nullptr;
    if (XR_FAILED(gipa(instance, name, &raw)) || !raw)
    {
        return false;
    }
    target = reinterpret_cast<Function>(raw);
    return true;
}

bool vkOk(VkResult result)
{
    return result == VK_SUCCESS;
}

bool releaseImageResources(CameraSession& camera)
{
    if (camera.device)
    {
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
        if (camera.cudaDevice >= 0)
        {
            // Failed completion quarantines both the borrowed input and our
            // external allocation. Never free it merely because teardown ran.
            if (cudaSetDevice(camera.cudaDevice) != cudaSuccess ||
                (camera.cudaPending && cudaStreamSynchronize(camera.cudaStream) != cudaSuccess))
            {
                camera.retired = true;
                return false;
            }
            camera.cudaPending = false;
        }
#endif
        if (camera.fence)
        {
            if (camera.fencePending)
            {
                // This helper owns only its copy, not every queue of the
                // borrowed device. Retain storage until its own fence proves
                // completion; device-wide idle is not our synchronization seam.
                if (vkWaitForFences(camera.device, 1, &camera.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
                {
                    isaaccapture::Logger::get("isaaccapture.ovxr_pip_layer.CameraSession")
                        ->error("IsaacTeleop PiP: GPU completion unproved; retaining upload allocations");
                    camera.retired = true;
                    camera.ready = false;
                    return false;
                }
            }
            vkDestroyFence(camera.device, camera.fence, nullptr);
            camera.fence = VK_NULL_HANDLE;
        }
        if (camera.pool)
        {
            vkDestroyCommandPool(camera.device, camera.pool, nullptr);
            camera.pool = VK_NULL_HANDLE;
            camera.command = VK_NULL_HANDLE;
        }
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
        if (camera.cudaPixels)
        {
            if (cudaFree(camera.cudaPixels) != cudaSuccess)
            {
                camera.retired = true;
                return false;
            }
            camera.cudaPixels = nullptr;
        }
        if (camera.cudaMemory)
        {
            if (cudaDestroyExternalMemory(camera.cudaMemory) != cudaSuccess)
            {
                camera.retired = true;
                return false;
            }
            camera.cudaMemory = nullptr;
        }
        if (camera.cudaStream)
        {
            if (cudaStreamDestroy(camera.cudaStream) != cudaSuccess)
            {
                camera.retired = true;
                return false;
            }
            camera.cudaStream = nullptr;
        }
        camera.cudaDevice = -1;
#endif
        if (camera.pixels)
        {
            vkUnmapMemory(camera.device, camera.memory);
        }
        if (camera.staging)
        {
            vkDestroyBuffer(camera.device, camera.staging, nullptr);
        }
        if (camera.memory)
        {
            vkFreeMemory(camera.device, camera.memory, nullptr);
        }
    }
    camera.fence = VK_NULL_HANDLE;
    camera.pool = VK_NULL_HANDLE;
    camera.command = VK_NULL_HANDLE;
    camera.pixels = nullptr;
    camera.staging = VK_NULL_HANDLE;
    camera.memory = VK_NULL_HANDLE;
    camera.fencePending = false;
    camera.ready = false;
    return true;
}

bool releaseSessionResources(CameraSession& camera, const Dispatch& dispatch)
{
    if (!releaseImageResources(camera))
    {
        return false;
    }
    if (camera.swapchain)
    {
        // OpenXR destruction requires completed application graphics work, not
        // reacquiring/releasing a runtime image. Our owned copies retired above.
        if (XR_FAILED(dispatch.destroySwapchain(camera.swapchain)))
        {
            return false;
        }
        camera.swapchain = XR_NULL_HANDLE;
    }
    camera.acquired = camera.waited = false;
    for (auto* space : { &camera.view, &camera.local })
    {
        if (*space && XR_FAILED(dispatch.destroySpace(*space)))
        {
            return false;
        }
        *space = XR_NULL_HANDLE;
    }
    camera.images.clear();
    return true;
}

bool createSwapchain(CameraSession& camera, const Dispatch& dispatch, uint32_t width, uint32_t height)
{
    if (camera.acquired || camera.retired)
    {
        return false;
    }
    if (!releaseImageResources(camera))
    {
        return false;
    }
    if (camera.swapchain)
    {
        if (XR_FAILED(dispatch.destroySwapchain(camera.swapchain)))
        {
            camera.retired = true;
            camera.ready = false;
            return false; // Retain the old handle for explicit owner cleanup.
        }
        camera.swapchain = XR_NULL_HANDLE;
        camera.images.clear();
    }
    // A new handle is not usable until enumeration succeeds. Invalidate the
    // old extent so a failed resize cannot admit a later old-size upload.
    camera.width = camera.height = 0;
    uint32_t count = 0;
    if (XR_FAILED(dispatch.formats(camera.session, 0, &count, nullptr)) || !count)
    {
        return false;
    }
    std::vector<int64_t> formats(count);
    if (XR_FAILED(dispatch.formats(camera.session, count, &count, formats.data())))
    {
        return false;
    }
    auto chosen = std::find(formats.begin(), formats.end(), VK_FORMAT_R8G8B8A8_UNORM);
    if (chosen == formats.end())
    {
        return false; // No color conversion or unverified swapchain format fallback.
    }
    XrSwapchainCreateInfo info{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
    info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    info.format = *chosen;
    info.sampleCount = 1;
    info.width = width;
    info.height = height;
    info.faceCount = 1;
    info.arraySize = 1;
    info.mipCount = 1;
    if (XR_FAILED(dispatch.createSwapchain(camera.session, &info, &camera.swapchain)))
    {
        return false;
    }
    count = 0;
    if (XR_FAILED(dispatch.images(camera.swapchain, 0, &count, nullptr)) || !count)
    {
        return false;
    }
    camera.images.resize(count);
    for (auto& image : camera.images)
    {
        image = {};
        image.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR;
    }
    if (XR_FAILED(dispatch.images(
            camera.swapchain, count, &count, reinterpret_cast<XrSwapchainImageBaseHeader*>(camera.images.data()))))
    {
        return false;
    }
    camera.width = width;
    camera.height = height;
    return true;
}

bool createUploadResources(CameraSession& camera, bool gpu = false)
{
    const auto fail = [&]()
    {
        releaseImageResources(camera); // No GPU submission exists yet.
        return false;
    };
    const VkDeviceSize length = static_cast<VkDeviceSize>(camera.width) * camera.height * 4;
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    VkExternalMemoryBufferCreateInfo external{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (gpu)
    {
        bufferInfo.pNext = &external;
    }
#endif
    bufferInfo.size = length;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!vkOk(vkCreateBuffer(camera.device, &bufferInfo, nullptr, &camera.staging)))
    {
        return fail();
    }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(camera.device, camera.staging, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(camera.physical, &properties);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
    {
        const VkMemoryPropertyFlags flags =
            gpu ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT :
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((requirements.memoryTypeBits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags)
        {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX)
    {
        return fail();
    }
    VkMemoryAllocateInfo allocation{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    VkMemoryDedicatedAllocateInfo dedicated{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    dedicated.buffer = camera.staging;
    VkExportMemoryAllocateInfo exportInfo{ VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
    exportInfo.pNext = &dedicated;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (gpu)
    {
        allocation.pNext = &exportInfo;
    }
#endif
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    if (!vkOk(vkAllocateMemory(camera.device, &allocation, nullptr, &camera.memory)) ||
        !vkOk(vkBindBufferMemory(camera.device, camera.staging, camera.memory, 0)) ||
        (!gpu && !vkOk(vkMapMemory(camera.device, camera.memory, 0, length, 0, &camera.pixels))))
    {
        return fail();
    }
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    if (gpu)
    {
        const auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(camera.device, "vkGetMemoryFdKHR"));
        VkMemoryGetFdInfoKHR fdInfo{ VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
        fdInfo.memory = camera.memory;
        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        if (!getFd || getFd(camera.device, &fdInfo, &fd) != VK_SUCCESS)
        {
            return fail();
        }
        cudaExternalMemoryHandleDesc memory{};
        memory.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memory.handle.fd = fd;
        memory.size = requirements.size;
        memory.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&camera.cudaMemory, &memory) != cudaSuccess)
        {
            close(fd); // Successful import alone transfers fd ownership to CUDA.
            return fail();
        }
        cudaExternalMemoryBufferDesc buffer{};
        buffer.size = length;
        if (cudaExternalMemoryGetMappedBuffer(&camera.cudaPixels, camera.cudaMemory, &buffer) != cudaSuccess ||
            cudaStreamCreateWithFlags(&camera.cudaStream, cudaStreamNonBlocking) != cudaSuccess)
        {
            return fail();
        }
    }
#endif
    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = camera.queueFamily;
    if (!vkOk(vkCreateCommandPool(camera.device, &poolInfo, nullptr, &camera.pool)))
    {
        return fail();
    }
    VkCommandBufferAllocateInfo commandInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    commandInfo.commandPool = camera.pool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (!vkOk(vkAllocateCommandBuffers(camera.device, &commandInfo, &camera.command)) ||
        !vkOk(vkCreateFence(camera.device, &fenceInfo, nullptr, &camera.fence)))
    {
        return fail();
    }
    return true;
}

bool upload(CameraSession& camera, const Dispatch& dispatch, const void* rgba)
{
    camera.ready = false;
    if (camera.retired)
    {
        return false;
    }
    if (camera.fencePending)
    {
        if (!vkOk(vkWaitForFences(camera.device, 1, &camera.fence, VK_TRUE, UINT64_MAX)))
        {
            camera.retired = true;
            return false;
        }
        camera.fencePending = false;
    }
    const VkDeviceSize length = static_cast<VkDeviceSize>(camera.width) * camera.height * 4;
    if (rgba)
    {
        std::memcpy(camera.pixels, rgba, static_cast<size_t>(length));
    }
    if (!camera.acquired)
    {
        XrSwapchainImageAcquireInfo acquireInfo{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
        if (dispatch.acquire(camera.swapchain, &acquireInfo, &camera.acquiredIndex) != XR_SUCCESS)
        {
            return false;
        }
        camera.acquired = true;
    }
    const auto release = [&]() -> bool
    {
        if (camera.waited)
        {
            XrSwapchainImageReleaseInfo info{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
            if (dispatch.release(camera.swapchain, &info) != XR_SUCCESS)
            {
                camera.retired = true;
                return false;
            }
            camera.waited = false;
            camera.acquired = false;
        }
        return true;
    };
    if (!camera.waited)
    {
        XrSwapchainImageWaitInfo waitInfo{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
        waitInfo.timeout = XR_INFINITE_DURATION;
        const XrResult waited = dispatch.wait(camera.swapchain, &waitInfo);
        if (waited == XR_TIMEOUT_EXPIRED)
        {
            return false; // Retry this same acquisition on the next publication.
        }
        if (waited != XR_SUCCESS)
        {
            camera.retired = true;
            return false;
        }
        camera.waited = true;
    }
    if (camera.acquiredIndex >= camera.images.size())
    {
        release();
        camera.retired = true;
        return false;
    }
    if (!vkOk(vkResetFences(camera.device, 1, &camera.fence)) || !vkOk(vkResetCommandPool(camera.device, camera.pool, 0)))
    {
        release();
        return false;
    }
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (!vkOk(vkBeginCommandBuffer(camera.command, &begin)))
    {
        release();
        return false;
    }
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    VkBufferMemoryBarrier fromCuda{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    if (camera.cudaPixels)
    {
        fromCuda.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        fromCuda.dstQueueFamilyIndex = camera.queueFamily;
        fromCuda.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        fromCuda.buffer = camera.staging;
        fromCuda.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(camera.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 1, &fromCuda, 0, nullptr);
    }
#endif
    const VkImage image = camera.images[camera.acquiredIndex].image;
    VkImageMemoryBarrier toTransfer{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = image;
    toTransfer.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    toTransfer.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(camera.command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toTransfer);
    VkBufferImageCopy copy{};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { camera.width, camera.height, 1 };
    vkCmdCopyBufferToImage(camera.command, camera.staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    if (camera.cudaPixels)
    {
        VkBufferMemoryBarrier toCuda = fromCuda;
        toCuda.srcQueueFamilyIndex = camera.queueFamily;
        toCuda.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        toCuda.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toCuda.dstAccessMask = 0;
        vkCmdPipelineBarrier(camera.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                             nullptr, 1, &toCuda, 0, nullptr);
    }
#endif
    VkImageMemoryBarrier toColor = toTransfer;
    toColor.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toColor.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    vkCmdPipelineBarrier(camera.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toColor);
    if (!vkOk(vkEndCommandBuffer(camera.command)))
    {
        release();
        return false;
    }
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &camera.command;
    if (!vkOk(vkQueueSubmit(camera.queue, 1, &submit, camera.fence)))
    {
        camera.retired = true;
        release();
        return false;
    }
    camera.fencePending = true;
    // Complete the bounded owned copy before the runtime reuses this image.
    if (!vkOk(vkWaitForFences(camera.device, 1, &camera.fence, VK_TRUE, UINT64_MAX)))
    {
        camera.retired = true;
        return false;
    }
    camera.fencePending = false;
    if (!release())
    {
        return false;
    }
    camera.ready = true;
    return true;
}

} // namespace

extern "C" PanelSession* isaacteleop_pip_create()
{
    return new (std::nothrow) PanelSession;
}

extern "C" uint32_t isaacteleop_pip_max_cameras()
{
    return kMaxCameras;
}
extern "C" uint32_t isaacteleop_pip_max_image_dimension()
{
    return kMaxImageDimension;
}

// Explicit application binding. All entry points are called on the OVXR owner
// thread, serialized with that context's queue and frame operations.
extern "C" int isaacteleop_pip_attach(PanelSession* owner,
                                      XrInstance instance,
                                      XrSession session,
                                      PFN_xrGetInstanceProcAddr gipa,
                                      VkPhysicalDevice physical,
                                      VkDevice device,
                                      VkQueue queue,
                                      uint32_t queueFamily)
{
    if (!instance || !session || !gipa || !physical || !device || !queue || !owner || owner->attached)
    {
        return -1;
    }
    Dispatch dispatch{};
    dispatch.gipa = gipa;
    if (!resolve(gipa, instance, "xrCreateReferenceSpace", dispatch.createSpace) ||
        !resolve(gipa, instance, "xrDestroySpace", dispatch.destroySpace) ||
        !resolve(gipa, instance, "xrEnumerateSwapchainFormats", dispatch.formats) ||
        !resolve(gipa, instance, "xrCreateSwapchain", dispatch.createSwapchain) ||
        !resolve(gipa, instance, "xrDestroySwapchain", dispatch.destroySwapchain) ||
        !resolve(gipa, instance, "xrEnumerateSwapchainImages", dispatch.images) ||
        !resolve(gipa, instance, "xrAcquireSwapchainImage", dispatch.acquire) ||
        !resolve(gipa, instance, "xrWaitSwapchainImage", dispatch.wait) ||
        !resolve(gipa, instance, "xrReleaseSwapchainImage", dispatch.release))
    {
        return -2;
    }
    // Register before allocation so even failed partial setup has an explicit
    // detach owner. Error -3 tells the binding to retain this cleanup obligation.
    owner->dispatch = dispatch;
    owner->attached = true;
    auto& cameras = owner->cameras;
    for (auto& camera : cameras)
    {
        camera.instance = instance;
        camera.session = session;
        camera.physical = physical;
        camera.device = device;
        camera.queue = queue;
        camera.queueFamily = queueFamily;
        XrReferenceSpaceCreateInfo space{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
        space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
        space.poseInReferenceSpace.orientation.w = 1.0f;
        if (XR_FAILED(dispatch.createSpace(session, &space, &camera.view)))
        {
            return -3;
        }
    }
    return 0;
}

extern "C" int isaacteleop_pip_destroy(PanelSession* owner)
{
    if (!owner)
        return 0;
    if (owner->attached)
    {
        for (auto& camera : owner->cameras)
        {
            if (!releaseSessionResources(camera, owner->dispatch))
                return -1; // Retain the owner until all borrowed reads retire.
        }
    }
    delete owner;
    return 0;
}

extern "C" const XrCompositionLayerBaseHeader* isaacteleop_pip_layer(PanelSession* owner, uint32_t cameraIndex)
{
    if (!owner || !owner->attached || cameraIndex >= kMaxCameras)
    {
        return nullptr;
    }
    const auto& camera = owner->cameras[cameraIndex];
    const auto& panel = owner->panels[cameraIndex];
    if (!camera.ready || camera.retired || !panel.enabled || !camera.swapchain ||
        !(panel.headLocked ? camera.view : camera.local))
    {
        return nullptr;
    }
    auto& quad = owner->quads[cameraIndex];
    quad = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    quad.layerFlags = 0; // Opaque camera panels, independent of projection alpha.
    quad.space = panel.headLocked ? camera.view : camera.local;
    quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    quad.subImage.swapchain = camera.swapchain;
    quad.subImage.imageRect.extent = { static_cast<int32_t>(camera.width), static_cast<int32_t>(camera.height) };
    quad.pose = panel.pose;
    quad.size = panel.size;
    return reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad);
}

extern "C" int isaacteleop_pip_configure(
    PanelSession* owner, uint32_t cameraIndex, int enabled, int headLocked, const float* pose7, const float* size2)
{
    if (!owner || cameraIndex >= kMaxCameras || !pose7 || !size2 || !std::isfinite(size2[0]) ||
        !std::isfinite(size2[1]) || size2[0] <= 0 || size2[1] <= 0)
    {
        return -1;
    }
    float orientationLength = 0.0f;
    for (int i = 0; i < 7; ++i)
    {
        if (!std::isfinite(pose7[i]))
        {
            return -1;
        }
        if (i >= 3)
        {
            orientationLength += pose7[i] * pose7[i];
        }
    }
    if (orientationLength < 0.99f || orientationLength > 1.01f)
    {
        return -1;
    }
    auto& panel = owner->panels[cameraIndex];
    panel.enabled = enabled != 0;
    panel.headLocked = headLocked != 0;
    panel.pose = { { pose7[3], pose7[4], pose7[5], pose7[6] }, { pose7[0], pose7[1], pose7[2] } };
    panel.size = { size2[0], size2[1] };
    return 0;
}

extern "C" int isaacteleop_pip_publish_rgba(
    PanelSession* owner, uint32_t cameraIndex, const void* rgba, uint32_t width, uint32_t height, size_t length)
{
    if (!owner || cameraIndex >= kMaxCameras || !rgba || !width || !height || width > kMaxImageDimension ||
        height > kMaxImageDimension || static_cast<uint64_t>(width) * height * 4 != length)
    {
        return UploadResult::InvalidArgument;
    }
    if (!owner->attached)
    {
        return UploadResult::NotAttached;
    }
    CameraSession& camera = owner->cameras[cameraIndex];
    const auto& panel = owner->panels[cameraIndex];
    if (!camera.device || !camera.queue || camera.retired || !(panel.headLocked ? camera.view : camera.local))
    {
        return UploadResult::CameraUnavailable;
    }
    const Dispatch& dispatch = owner->dispatch;
    if ((camera.width != width || camera.height != height || !camera.swapchain) &&
        !createSwapchain(camera, dispatch, width, height))
    {
        return UploadResult::SwapchainFailed;
    }
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    if (camera.cudaDevice >= 0)
    {
        return UploadResult::UploadModeMismatch; // Do not replace live CUDA aliases with CPU staging resources.
    }
#endif
    if ((!camera.staging || !camera.memory || !camera.pixels || !camera.pool || !camera.command || !camera.fence) &&
        !createUploadResources(camera))
    {
        return UploadResult::AllocationFailed;
    }
    return upload(camera, dispatch, rgba) ? UploadResult::Success : UploadResult::UploadFailed;
}


// Application API, not an OVXR export. The caller owns a completed OVRTX map
// until this synchronous upload returns. CompletionUnproved retains that map
// until completion is proved; the affected panel retires.
extern "C" int isaacteleop_pip_supports_cuda()
{
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    return 1;
#else
    return 0;
#endif
}

#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
namespace
{
cudaError_t copyCameraArray(CameraSession& camera, const void* array, uint32_t width, uint32_t height, int sourceDevice)
{
    if (sourceDevice == camera.cudaDevice)
    {
        return cudaMemcpy2DFromArrayAsync(camera.cudaPixels, static_cast<size_t>(width) * 4,
                                          static_cast<cudaArray_const_t>(array), 0, 0, static_cast<size_t>(width) * 4,
                                          height, cudaMemcpyDeviceToDevice, camera.cudaStream);
    }
    // RenderProducts can be assigned to a different GPU from the XR binding.
    // Preserve their assignment; copy to the binding's device, not host pixels.
    cudaMemcpy3DPeerParms copy{};
    copy.srcArray = const_cast<cudaArray_t>(static_cast<cudaArray_const_t>(array));
    copy.srcDevice = sourceDevice;
    copy.dstPtr = { camera.cudaPixels, static_cast<size_t>(width) * 4, static_cast<size_t>(width) * 4, height };
    copy.dstDevice = camera.cudaDevice;
    // With a CUDA array, extent.width is in array elements (RGBA pixels).
    copy.extent = { width, height, 1 };
    return cudaMemcpy3DPeerAsync(&copy, camera.cudaStream);
}

int copyCameraArrayAndWait(CameraSession& camera, const void* array, uint32_t width, uint32_t height, int sourceDevice)
{
    camera.cudaPending = true;
    const auto copied = copyCameraArray(camera, array, width, height, sourceDevice);
    // CUDA calls may surface an earlier asynchronous error. Even an enqueue
    // failure must prove that this stream has stopped reading the borrowed map.
    if (cudaStreamSynchronize(camera.cudaStream) != cudaSuccess)
    {
        camera.retired = true;
        return UploadResult::CompletionUnproved;
    }
    camera.cudaPending = false;
    return copied == cudaSuccess ? UploadResult::Success : UploadResult::UploadFailed;
}
} // namespace
#endif

// Completion-only recovery never retries a copy or destroys native resources.
extern "C" int isaacteleop_pip_wait_cuda(PanelSession* owner, uint32_t cameraIndex)
{
    if (!owner || cameraIndex >= kMaxCameras)
        return UploadResult::InvalidArgument;
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    auto& camera = owner->cameras[cameraIndex];
    if (camera.cudaPending)
    {
        if (cudaSetDevice(camera.cudaDevice) != cudaSuccess || cudaStreamSynchronize(camera.cudaStream) != cudaSuccess)
            return UploadResult::CompletionUnproved;
        camera.cudaPending = false;
    }
#endif
    return UploadResult::Success;
}

extern "C" int isaacteleop_pip_publish_cuda_array(
    PanelSession* owner, uint32_t cameraIndex, const void* array, uint32_t width, uint32_t height, int device)
{
#if defined(ISAAC_TELEOP_PIP_CUDA) && defined(__linux__)
    if (!owner || cameraIndex >= kMaxCameras || !array || !width || !height || width > kMaxImageDimension ||
        height > kMaxImageDimension || device < 0)
    {
        return UploadResult::InvalidArgument;
    }
    if (!owner->attached)
    {
        return UploadResult::NotAttached;
    }
    CameraSession& camera = owner->cameras[cameraIndex];
    if (!camera.device || !camera.queue || camera.retired)
    {
        return UploadResult::CameraUnavailable;
    }
    // Resolve the borrowed Vulkan device once. A camera output's CUDA ordinal
    // is a source identity, not proof it is the destination Vulkan device.
    if (camera.vulkanCudaDevice < 0)
    {
        VkPhysicalDeviceIDProperties id{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        properties.pNext = &id;
        vkGetPhysicalDeviceProperties2(camera.physical, &properties);
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess)
        {
            return UploadResult::CudaUnavailable;
        }
        for (int candidate = 0; candidate < count; ++candidate)
        {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, candidate) == cudaSuccess &&
                std::memcmp(id.deviceUUID, properties.uuid.bytes, VK_UUID_SIZE) == 0)
            {
                camera.vulkanCudaDevice = candidate;
                break;
            }
        }
        if (camera.vulkanCudaDevice < 0)
        {
            return UploadResult::CudaUnavailable;
        }
    }
    if (cudaSetDevice(camera.vulkanCudaDevice) != cudaSuccess)
    {
        return UploadResult::CudaUnavailable;
    }
    const Dispatch& dispatch = owner->dispatch;
    if ((camera.width != width || camera.height != height || !camera.swapchain) &&
        !createSwapchain(camera, dispatch, width, height))
    {
        return UploadResult::SwapchainFailed;
    }
    if (!camera.cudaPixels || !camera.pool || !camera.command || !camera.fence)
    {
        if (!releaseImageResources(camera))
        {
            return UploadResult::CompletionUnproved;
        }
        camera.cudaDevice = camera.vulkanCudaDevice;
        if (!createUploadResources(camera, true))
        {
            return UploadResult::AllocationFailed;
        }
    }
    if (camera.fencePending)
    {
        if (vkWaitForFences(camera.device, 1, &camera.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        {
            camera.retired = true;
            return UploadResult::UploadFailed;
        }
        camera.fencePending = false;
    }
    camera.ready = false;
    const int copied = copyCameraArrayAndWait(camera, array, width, height, device);
    if (copied != UploadResult::Success)
    {
        return copied;
    }
    // No external semaphore wait is introduced: preserve the current host-wait
    // scheduling policy. This is a GPU-only copy, not end-to-end zero-copy.
    return upload(camera, dispatch, nullptr) ? UploadResult::Success : UploadResult::UploadFailed;
#else
    return UploadResult::CudaUnavailable;
#endif
}
