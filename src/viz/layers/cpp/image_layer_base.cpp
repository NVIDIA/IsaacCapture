// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/viz/layers/image_layer_base.hpp"

#include <viz/core/cuda_error.hpp>
#include <viz/core/vk_context.hpp>

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace viz
{

namespace
{

// viz_device: the device the layer's interop images live on — lets a
// cross-device failure say so instead of just "invalid argument"
// (viz/core/cuda_error.hpp).
void check_cuda(cudaError_t result, const char* layer_type, const char* what, int viz_device)
{
    viz::check_cuda(result, layer_type, what, viz_device);
}

// Queue an async D2D copy of ``buf`` → ``image.cuda_array()`` on
// ``stream``. Shared between the mono and stereo submit paths.
void enqueue_copy(const VizBuffer& buf, DeviceImage& image, cudaStream_t stream, const std::string& layer_type, int viz_device)
{
    const size_t row_bytes = static_cast<size_t>(buf.width) * bytes_per_pixel(buf.format);
    const size_t src_pitch = (buf.pitch == 0) ? row_bytes : buf.pitch;
    const cudaError_t err = cudaMemcpy2DToArrayAsync(
        image.cuda_array(), 0, 0, buf.data, src_pitch, row_bytes, buf.height, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cuda_error_message(layer_type, "submit: cudaMemcpy2DToArrayAsync", err, viz_device));
    }
}

} // namespace

ImageLayerBase::ImageLayerBase(const VkContext& ctx,
                               std::string layer_type,
                               std::string name,
                               Resolution resolution,
                               PixelFormat format,
                               bool stereo,
                               uint32_t mip_levels)
    : LayerBase(std::move(name)),
      ctx_(&ctx),
      layer_type_(std::move(layer_type)),
      resolution_(resolution),
      format_(format),
      stereo_(stereo)
{
    // Composition (draw sampler / swapchain copy) assumes a color image;
    // depth views aren't color-samplable or copy-compatible with the
    // color swapchains.
    if (format_ != PixelFormat::kRGBA8)
    {
        throw std::invalid_argument(layer_type_ + ": only PixelFormat::kRGBA8 is supported");
    }
    if (resolution_.width == 0 || resolution_.height == 0)
    {
        throw std::invalid_argument(layer_type_ + ": resolution must be non-zero");
    }
    if (!ctx.is_initialized())
    {
        throw std::invalid_argument(layer_type_ + ": VkContext is not initialized");
    }

    // Atomic<uint8_t>'s default state is unspecified per the standard;
    // explicitly seed every entry to kSlotNone so submit / promote /
    // get_wait_semaphores see a defined initial state.
    for (auto& e : in_use_)
    {
        e.store(kSlotNone, std::memory_order_relaxed);
    }
    last_in_use_slot_.store(kSlotNone, std::memory_order_relaxed);

    // DeviceImages are RAII — a mid-loop throw releases the already-
    // created slots via the members' destructors.
    for (auto& slot : slots_)
    {
        slot = DeviceImage::create(*ctx_, resolution_, format_, mip_levels);
    }
    if (stereo_)
    {
        for (auto& slot : slots_right_)
        {
            slot = DeviceImage::create(*ctx_, resolution_, format_, mip_levels);
        }
    }
}

ImageLayerBase::~ImageLayerBase()
{
    destroy_images();
}

void ImageLayerBase::destroy_images()
{
    for (auto& slot : slots_)
    {
        slot.reset();
    }
    for (auto& slot : slots_right_)
    {
        slot.reset();
    }
    latest_.store(kSlotNone, std::memory_order_release);
    for (auto& e : in_use_)
    {
        e.store(kSlotNone, std::memory_order_release);
    }
    last_in_use_slot_.store(kSlotNone, std::memory_order_release);
}

void ImageLayerBase::require_alive(const char* what) const
{
    if (!slots_[0])
    {
        throw std::logic_error(layer_type_ + "::" + what + " called after destroy()");
    }
}

const DeviceImage* ImageLayerBase::device_image(uint32_t slot) const noexcept
{
    if (slot >= kSlotCount)
    {
        return nullptr;
    }
    return slots_[slot].get();
}

const DeviceImage* ImageLayerBase::device_image_right(uint32_t slot) const noexcept
{
    if (slot >= kSlotCount)
    {
        return nullptr;
    }
    return slots_right_[slot].get();
}

uint8_t ImageLayerBase::pick_free_slot(uint8_t latest) const noexcept
{
    // Forbidden = {latest} ∪ in_use_. kSlotCount = kMaxFramesInFlight + 2
    // guarantees one free slot under the invariant; we still return
    // kSlotNone defensively if the invariant ever breaks, so submit()
    // drops the publish rather than overwriting a slot the GPU is sampling.
    static_assert(kSlotCount > kMaxFramesInFlight + 1,
                  "kSlotCount must exceed kMaxFramesInFlight + 1 so at least one slot is free");
    for (uint8_t i = 0; i < kSlotCount; ++i)
    {
        if (i == latest)
            continue;
        bool conflicts = false;
        for (uint32_t k = 0; k < kMaxFramesInFlight; ++k)
        {
            if (i == in_use_[k].load(std::memory_order_acquire))
            {
                conflicts = true;
                break;
            }
        }
        if (!conflicts)
        {
            return i;
        }
    }
    return kSlotNone;
}

void ImageLayerBase::validate_submit_buffer(const VizBuffer& buf, const char* label) const
{
    if (buf.space != MemorySpace::kDevice)
    {
        throw std::invalid_argument(layer_type_ + "::submit: " + label + " must be MemorySpace::kDevice");
    }
    if (buf.width != resolution_.width || buf.height != resolution_.height)
    {
        throw std::invalid_argument(layer_type_ + "::submit: " + label + " dimensions do not match layer resolution");
    }
    if (buf.format != format_)
    {
        throw std::invalid_argument(layer_type_ + "::submit: " + label + " format does not match layer format");
    }
    if (buf.data == nullptr)
    {
        throw std::invalid_argument(layer_type_ + "::submit: " + label + ".data is null");
    }
}

void ImageLayerBase::submit(const VizBuffer& src, cudaStream_t stream)
{
    require_alive("submit");
    if (stereo_)
    {
        throw std::logic_error(layer_type_ +
                               "::submit: this layer is stereo — use the two-arg submit(left, right) overload");
    }
    validate_submit_buffer(src, "src");

    const uint8_t latest = latest_.load(std::memory_order_acquire);
    const uint8_t slot = pick_free_slot(latest);
    if (slot == kSlotNone)
    {
        // Mailbox drop: producer outran the renderer beyond the sizing
        // invariant. Keep latest_ where it is; consumer keeps using it.
        return;
    }
    DeviceImage& image = *slots_[slot];

    check_cuda(cudaSetDevice(ctx_->cuda_device_id()), layer_type_.c_str(), "cudaSetDevice", ctx_->cuda_device_id());
    enqueue_copy(src, image, stream, layer_type_, ctx_->cuda_device_id());
    image.cuda_signal_write_done(stream);

    // Wait for the D2D copy to complete before returning. Sources publish
    // buffers by reference and treat ``latest()`` returning them as proof
    // of consumption; without a sync here a fast producer could wrap the
    // mailbox and overwrite src.data while our async memcpy is still
    // reading from it. Cost is ~0.5 ms per 1080p submit on the caller's
    // thread; the render path is unaffected.
    // Borrowed CUDA IPC frames remain mapped until this copy completes.
    check_cuda(
        cudaStreamSynchronize(stream), layer_type_.c_str(), "cudaStreamSynchronize(submit)", ctx_->cuda_device_id());

    // memory_order_release pairs with the renderer's acquire load.
    latest_.store(slot, std::memory_order_release);
}

void ImageLayerBase::submit(const VizBuffer& left, const VizBuffer& right, cudaStream_t stream)
{
    require_alive("submit");
    if (!stereo_)
    {
        throw std::logic_error(layer_type_ + "::submit: this layer is mono — call submit(src) with a single buffer");
    }
    validate_submit_buffer(left, "left");
    validate_submit_buffer(right, "right");

    const uint8_t latest = latest_.load(std::memory_order_acquire);
    const uint8_t slot = pick_free_slot(latest);
    if (slot == kSlotNone)
    {
        return;
    }
    DeviceImage& image_l = *slots_[slot];
    DeviceImage& image_r = *slots_right_[slot];

    check_cuda(cudaSetDevice(ctx_->cuda_device_id()), layer_type_.c_str(), "cudaSetDevice", ctx_->cuda_device_id());
    // Both copies on the same stream + a single signal on the left's
    // semaphore. Stream ordering guarantees the right copy completes
    // before the signal fires, so the renderer waiting on the left's
    // semaphore implies the right is ready too. No second semaphore
    // needed — by construction the renderer cannot see a half-pair.
    //
    // Stream precondition (see header): ``left.data`` and ``right.data``
    // must both be reachable from ``stream`` by the time control reaches
    // here. If a producer wrote either buffer on a different stream, the
    // caller is responsible for syncing it before submit; otherwise the
    // memcpy below may read pre-write state on that eye.
    enqueue_copy(left, image_l, stream, layer_type_, ctx_->cuda_device_id());
    enqueue_copy(right, image_r, stream, layer_type_, ctx_->cuda_device_id());
    image_l.cuda_signal_write_done(stream);

    check_cuda(cudaStreamSynchronize(stream), layer_type_.c_str(), "cudaStreamSynchronize(submit-stereo)",
               ctx_->cuda_device_id());

    latest_.store(slot, std::memory_order_release);
}

uint8_t ImageLayerBase::promote_slot(uint32_t in_flight_slot)
{
    // Backends are contracted to image_count <= kMaxFramesInFlight; if
    // that ever breaks, two in-flight frames would alias on the same
    // in_use_ entry and we'd lose the slot-tracking invariant.
    if (in_flight_slot >= kMaxFramesInFlight)
    {
        throw std::logic_error(layer_type_ + ": in_flight_slot " + std::to_string(in_flight_slot) +
                               " >= kMaxFramesInFlight (" + std::to_string(kMaxFramesInFlight) +
                               "); bump kMaxFramesInFlight to match the backend's image_count");
    }

    // Promote latest_ -> in_use_[in_flight_slot]. The compositor's
    // per-slot fence wait at the top of render() guarantees the GPU
    // has finished sampling the previous in_use_ value. The consuming
    // call (draw record or native-layer copy) reads the same entry —
    // the promoting call and the consuming call MUST agree on
    // in_flight_slot.
    const uint8_t latest = latest_.load(std::memory_order_acquire);
    const uint32_t idx = in_flight_slot;
    if (latest != kSlotNone)
    {
        in_use_[idx].store(latest, std::memory_order_release);
    }
    const uint8_t cur = in_use_[idx].load(std::memory_order_acquire);
    if (cur != kSlotNone)
    {
        // Record which slot this frame is sampling so get_wait_semaphores
        // (called by the compositor before submit) reads the matching
        // cuda_done_writing semaphore.
        last_in_use_slot_.store(cur, std::memory_order_release);
    }
    return cur;
}

uint8_t ImageLayerBase::promoted_slot(uint32_t in_flight_slot) const noexcept
{
    if (in_flight_slot >= kMaxFramesInFlight)
    {
        return kSlotNone;
    }
    return in_use_[in_flight_slot].load(std::memory_order_acquire);
}

std::vector<LayerBase::WaitSemaphore> ImageLayerBase::get_wait_semaphores() const
{
    // The compositor promotes first (record_pre_render_pass or the
    // native-layer acquire, both of which set last_in_use_slot_). We
    // return THAT slot's cuda_done_writing semaphore so the submit waits
    // for the producer's memcpy, gated at the subclass's first read.
    const uint8_t cur = last_in_use_slot_.load(std::memory_order_acquire);
    if (cur == kSlotNone || !slots_[cur])
    {
        return {};
    }
    const DeviceImage& image = *slots_[cur];
    const uint64_t value = image.cuda_done_writing_value();
    if (value == 0)
    {
        return {};
    }
    return {
        WaitSemaphore{
            image.cuda_done_writing(),
            value,
            first_read_stage(),
        },
    };
}

} // namespace viz
