// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Write-attempt markers for the CUDA IPC pixel slots that camera_receiver (the slot writer) shares with its readers.
// The writer advances a slot's marker before it touches the slot's pixels; a reader loads it again after its copy
// has finished and accepts the copy only if the marker still equals the generation the frame notice named.
//
// The operations are sequentially consistent process-shared atomics. Their ordering against GPU work is the caller's
// duty: advance the marker before launching the first write, and load it only after the GPU reads have completed.
#pragma once

#include <sys/mman.h>
#include <sys/stat.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace camera_ipc
{

inline constexpr std::uint32_t kMarkerMagic = 0x314b4d53u; // "SMK1"
inline constexpr std::uint32_t kMarkerLayoutVersion = 1;
inline constexpr std::uint32_t kMaxStreams = 16;
inline constexpr std::uint32_t kMaxSlots = 32;
// A marker never wraps: a writer that reaches this restarts under a new region and epoch instead.
inline constexpr std::uint64_t kMaxGeneration = 1ull << 62;

static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "slot markers need lock-free 64-bit atomics");
static_assert(std::atomic<std::uint32_t>::is_always_lock_free, "slot markers need lock-free 32-bit atomics");

struct alignas(64) MarkerCell
{
    std::atomic<std::uint64_t> generation;
};

struct alignas(64) MarkerHeader
{
    std::uint32_t magic;
    std::uint32_t layout_version;
    std::uint64_t epoch;
    std::uint32_t total_size;
    std::uint32_t stream_count;
    std::uint32_t slot_count[kMaxStreams];
    std::atomic<std::uint32_t> ready;
};

struct MarkerRegion
{
    MarkerHeader header;
    MarkerCell cells[kMaxStreams][kMaxSlots];
};

class SlotMarkers
{
public:
    // Creates and initialises the region. The name is unique per call and the object is private to the user.
    static SlotMarkers create(const std::string& name_prefix, const std::vector<std::uint32_t>& slot_counts)
    {
        require_lock_free();
        if (slot_counts.empty() || slot_counts.size() > kMaxStreams)
        {
            throw std::invalid_argument("slot markers: stream count must be 1.." + std::to_string(kMaxStreams));
        }
        for (std::uint32_t count : slot_counts)
        {
            if (count < 1 || count > kMaxSlots)
            {
                throw std::invalid_argument("slot markers: slots per stream must be 1.." + std::to_string(kMaxSlots));
            }
        }

        std::random_device entropy;
        const std::uint64_t epoch = random_u64(entropy);
        char suffix[40];
        std::snprintf(suffix, sizeof(suffix), "_%d_%016llx", static_cast<int>(::getpid()),
                      static_cast<unsigned long long>(random_u64(entropy)));
        const std::string name = name_prefix + suffix;

        const int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0)
        {
            throw std::runtime_error("slot markers: shm_open(" + name + ") failed: " + std::strerror(errno));
        }
        if (::fchmod(fd, 0600) != 0 || ::ftruncate(fd, sizeof(MarkerRegion)) != 0)
        {
            const std::string what = std::strerror(errno);
            ::close(fd);
            ::shm_unlink(name.c_str());
            throw std::runtime_error("slot markers: sizing " + name + " failed: " + what);
        }
        void* mapped = ::mmap(nullptr, sizeof(MarkerRegion), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (mapped == MAP_FAILED)
        {
            const std::string what = std::strerror(errno);
            ::shm_unlink(name.c_str());
            throw std::runtime_error("slot markers: mmap(" + name + ") failed: " + what);
        }

        // The mapping is zero-filled; construct the atomics in place and publish `ready` last.
        auto* region = new (mapped) MarkerRegion();
        region->header.magic = kMarkerMagic;
        region->header.layout_version = kMarkerLayoutVersion;
        region->header.epoch = epoch;
        region->header.total_size = static_cast<std::uint32_t>(sizeof(MarkerRegion));
        region->header.stream_count = static_cast<std::uint32_t>(slot_counts.size());
        for (std::size_t i = 0; i < slot_counts.size(); ++i)
        {
            region->header.slot_count[i] = slot_counts[i];
        }
        region->header.ready.store(1);
        return SlotMarkers(region, name, true);
    }

    // Maps a region a writer advertised. The advertised size, layout version and epoch must all match the object.
    static SlotMarkers attach(const std::string& name, std::uint64_t size, std::uint32_t layout_version, std::uint64_t epoch)
    {
        require_lock_free();
        if (layout_version != kMarkerLayoutVersion || size != sizeof(MarkerRegion))
        {
            throw std::runtime_error("slot markers: unsupported layout (version " + std::to_string(layout_version) +
                                     ", size " + std::to_string(size) + ")");
        }
        const int fd = ::shm_open(name.c_str(), O_RDWR, 0);
        if (fd < 0)
        {
            throw std::runtime_error("slot markers: shm_open(" + name + ") failed: " + std::strerror(errno));
        }
        struct stat info
        {
        };
        if (::fstat(fd, &info) != 0 || static_cast<std::uint64_t>(info.st_size) != size)
        {
            ::close(fd);
            throw std::runtime_error("slot markers: " + name + " has the wrong size");
        }
        void* mapped = ::mmap(nullptr, sizeof(MarkerRegion), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (mapped == MAP_FAILED)
        {
            throw std::runtime_error("slot markers: mmap(" + name + ") failed: " + std::strerror(errno));
        }
        auto* region = static_cast<MarkerRegion*>(mapped);
        const MarkerHeader& header = region->header;
        bool sane = header.ready.load() == 1 && header.magic == kMarkerMagic &&
                    header.layout_version == kMarkerLayoutVersion && header.epoch == epoch &&
                    header.total_size == sizeof(MarkerRegion) && header.stream_count >= 1 &&
                    header.stream_count <= kMaxStreams;
        for (std::uint32_t i = 0; sane && i < header.stream_count; ++i)
        {
            sane = header.slot_count[i] >= 1 && header.slot_count[i] <= kMaxSlots;
        }
        if (!sane)
        {
            ::munmap(mapped, sizeof(MarkerRegion));
            throw std::runtime_error("slot markers: " + name + " is not the advertised region");
        }
        return SlotMarkers(region, name, false);
    }

    SlotMarkers(const SlotMarkers&) = delete;
    SlotMarkers& operator=(const SlotMarkers&) = delete;
    SlotMarkers(SlotMarkers&& other) noexcept
        : region_(std::exchange(other.region_, nullptr)), name_(std::move(other.name_)), owner_(other.owner_)
    {
    }
    SlotMarkers& operator=(SlotMarkers&& other) noexcept
    {
        if (this != &other)
        {
            close();
            region_ = std::exchange(other.region_, nullptr);
            name_ = std::move(other.name_);
            owner_ = other.owner_;
        }
        return *this;
    }
    ~SlotMarkers()
    {
        close();
    }

    // Unmaps the region; the creator also removes its name so no new reader can attach.
    void close() noexcept
    {
        if (region_ == nullptr)
        {
            return;
        }
        ::munmap(region_, sizeof(MarkerRegion));
        region_ = nullptr;
        if (owner_)
        {
            ::shm_unlink(name_.c_str());
        }
    }

    const std::string& name() const
    {
        return name_;
    }
    static constexpr std::uint64_t region_size()
    {
        return sizeof(MarkerRegion);
    }
    std::uint64_t epoch() const
    {
        return region().header.epoch;
    }
    std::uint32_t layout_version() const
    {
        return region().header.layout_version;
    }
    std::uint32_t stream_count() const
    {
        return region().header.stream_count;
    }
    std::uint32_t slot_count(std::uint32_t stream) const
    {
        check_stream(stream);
        return region().header.slot_count[stream];
    }

    // Writer: records a write attempt and returns its generation. Call before the first GPU operation that may touch
    // the slot's pixels, and keep the result to publish if the attempt succeeds. Failed attempts are never undone.
    std::uint64_t begin_write(std::uint32_t stream, std::uint32_t slot)
    {
        std::atomic<std::uint64_t>& marker = cell_at(stream, slot).generation;
        const std::uint64_t generation = marker.fetch_add(1) + 1;
        if (generation >= kMaxGeneration)
        {
            throw std::runtime_error("slot markers: generation space exhausted; restart under a new region");
        }
        // The increment is globally visible before any later launch, including device doorbell writes.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return generation;
    }

    // Reader: the slot's current generation. Load it only after every GPU read of the slot has completed.
    std::uint64_t load(std::uint32_t stream, std::uint32_t slot) const
    {
        return cell_at(stream, slot).generation.load();
    }

    // Reader: whether no write has begun since `generation` was announced.
    bool unchanged(std::uint32_t stream, std::uint32_t slot, std::uint64_t generation) const
    {
        return load(stream, slot) == generation;
    }

private:
    SlotMarkers(MarkerRegion* region, std::string name, bool owner)
        : region_(region), name_(std::move(name)), owner_(owner)
    {
    }

    static void require_lock_free()
    {
        std::atomic<std::uint64_t> probe64{ 0 };
        std::atomic<std::uint32_t> probe32{ 0 };
        if (!probe64.is_lock_free() || !probe32.is_lock_free())
        {
            throw std::runtime_error("slot markers: this platform has no lock-free process-shared atomics");
        }
    }

    static std::uint64_t random_u64(std::random_device& entropy)
    {
        std::uint64_t value = 0;
        while (value == 0)
        {
            value = (static_cast<std::uint64_t>(entropy()) << 32) ^ entropy();
        }
        return value;
    }

    void check_stream(std::uint32_t stream) const
    {
        if (stream >= region().header.stream_count)
        {
            throw std::out_of_range("slot markers: stream index out of range");
        }
    }

    MarkerCell& cell_at(std::uint32_t stream, std::uint32_t slot) const
    {
        check_stream(stream);
        if (slot >= region().header.slot_count[stream])
        {
            throw std::out_of_range("slot markers: slot index out of range");
        }
        return region().cells[stream][slot];
    }

    MarkerRegion& region() const
    {
        if (region_ == nullptr)
        {
            throw std::runtime_error("slot markers: the region is closed");
        }
        return *region_;
    }

    MarkerRegion* region_;
    std::string name_;
    bool owner_;
};

} // namespace camera_ipc
