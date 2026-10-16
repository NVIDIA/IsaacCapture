// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "preview.hpp"

#include "diagnostics.hpp"

#include <SDL.h>
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <libobsensor/ObSensor.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace plugins::ego
{
namespace
{

struct RgbFrame
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;
    int64_t device_timestamp_ns = 0;
    uint32_t capture_epoch = 0;
};

struct Texture
{
    SDL_Texture* handle = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

    ~Texture()
    {
        if (handle)
            SDL_DestroyTexture(handle);
    }

    void update(SDL_Renderer* renderer, const RgbFrame& image)
    {
        if (!handle || width != image.width || height != image.height)
        {
            if (handle)
                SDL_DestroyTexture(handle);
            handle = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
                                       static_cast<int>(image.width), static_cast<int>(image.height));
            width = image.width;
            height = image.height;
        }
        if (handle)
            SDL_UpdateTexture(handle, nullptr, image.pixels.data(), static_cast<int>(image.width * 3));
    }
};

class Decoder
{
public:
    explicit Decoder(core::EgoPixelFormat format) : format_(format)
    {
        if (format_ == core::EgoPixelFormat_Mjpg)
        {
            converter_ = std::make_unique<ob::FormatConvertFilter>();
            converter_->setFormatConvertType(FORMAT_MJPG_TO_RGB);
            return;
        }
        const auto codec_id = format_ == core::EgoPixelFormat_H264 ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC;
        const AVCodec* codec = avcodec_find_decoder(codec_id);
        if (!codec)
            throw std::runtime_error("FFmpeg decoder unavailable for Ego preview");
        try
        {
            codec_ = avcodec_alloc_context3(codec);
            if (!codec_)
                throw std::runtime_error("Unable to allocate FFmpeg Ego preview decoder");
            codec_->pkt_timebase = { 1, 1'000'000'000 };
            if (avcodec_open2(codec_, codec, nullptr) < 0)
                throw std::runtime_error("Unable to open FFmpeg Ego preview decoder");
            decoded_ = av_frame_alloc();
            packet_ = av_packet_alloc();
            if (!decoded_ || !packet_)
                throw std::runtime_error("Unable to allocate FFmpeg preview frame or packet");
        }
        catch (...)
        {
            av_packet_free(&packet_);
            av_frame_free(&decoded_);
            avcodec_free_context(&codec_);
            throw;
        }
    }

    ~Decoder()
    {
        sws_freeContext(sws_);
        av_packet_free(&packet_);
        av_frame_free(&decoded_);
        avcodec_free_context(&codec_);
    }

    bool decode(const CapturedFrame& input, RgbFrame& output)
    {
        if (format_ == core::EgoPixelFormat_Mjpg)
        {
            auto source = ob::FrameFactory::createVideoFrameFromBuffer(
                input.metadata.stream == core::EgoCameraStream_ColorLeft ? OB_FRAME_COLOR_LEFT : OB_FRAME_COLOR_RIGHT,
                OB_FORMAT_MJPG, input.metadata.width, input.metadata.height,
                const_cast<uint8_t*>(input.encoded_data.data()), [](uint8_t*) {},
                static_cast<uint32_t>(input.encoded_data.size()));
            // SDK conversion may reject a frame; keep the last valid preview image.
            const auto converted = converter_->process(source);
            if (!converted || !converted->is<ob::VideoFrame>())
                return false;
            const auto rgb = converted->as<ob::VideoFrame>();
            const uint64_t expected_size = static_cast<uint64_t>(input.metadata.width) * input.metadata.height * 3;
            if (rgb->getFormat() != OB_FORMAT_RGB || rgb->getWidth() != input.metadata.width ||
                rgb->getHeight() != input.metadata.height || rgb->getDataSize() != expected_size || !rgb->getData())
                return false;
            output.width = rgb->getWidth();
            output.height = rgb->getHeight();
            output.pixels.assign(rgb->getData(), rgb->getData() + rgb->getDataSize());
            output.device_timestamp_ns = input.sample_time_raw_device_clock_ns;
            return true;
        }

        if (input.encoded_data.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Ego preview packet exceeds FFmpeg's size limit");
        av_packet_unref(packet_);
        // Refcounted packets provide FFmpeg's required zero padding and retain buffer ownership.
        if (av_new_packet(packet_, static_cast<int>(input.encoded_data.size())) < 0)
            throw std::runtime_error("Unable to allocate FFmpeg preview packet data");
        std::memcpy(packet_->data, input.encoded_data.data(), input.encoded_data.size());
        packet_->pts = input.sample_time_raw_device_clock_ns;
        bool produced = false;
        int status = avcodec_send_packet(codec_, packet_);
        if (status == AVERROR(EAGAIN))
        {
            produced = receive(input, output);
            status = avcodec_send_packet(codec_, packet_);
        }
        if (status < 0)
            throw std::runtime_error("FFmpeg rejected an accepted Ego preview access unit");
        av_packet_unref(packet_);
        return receive(input, output) || produced;
    }

private:
    bool receive(const CapturedFrame& input, RgbFrame& output)
    {
        bool produced = false;
        while (true)
        {
            const int status = avcodec_receive_frame(codec_, decoded_);
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF)
                return produced;
            if (status < 0 || decoded_->decode_error_flags != 0 ||
                decoded_->width != static_cast<int>(input.metadata.width) ||
                decoded_->height != static_cast<int>(input.metadata.height))
                throw std::runtime_error("FFmpeg could not decode a valid Ego preview image");
            output.width = static_cast<uint32_t>(decoded_->width);
            output.height = static_cast<uint32_t>(decoded_->height);
            output.pixels.resize(static_cast<size_t>(output.width) * output.height * 3);
            output.device_timestamp_ns = decoded_->best_effort_timestamp;
            sws_ = sws_getCachedContext(sws_, decoded_->width, decoded_->height,
                                        static_cast<AVPixelFormat>(decoded_->format), decoded_->width, decoded_->height,
                                        AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            if (!sws_)
                throw std::runtime_error("Unable to allocate Ego preview RGB converter");
            uint8_t* destinations[] = { output.pixels.data() };
            int strides[] = { decoded_->width * 3 };
            if (sws_scale(sws_, decoded_->data, decoded_->linesize, 0, decoded_->height, destinations, strides) !=
                decoded_->height)
                throw std::runtime_error("Unable to convert Ego preview image to RGB");
            produced = true;
        }
    }

    core::EgoPixelFormat format_;
    std::unique_ptr<ob::FormatConvertFilter> converter_;
    AVCodecContext* codec_ = nullptr;
    AVFrame* decoded_ = nullptr;
    AVPacket* packet_ = nullptr;
    SwsContext* sws_ = nullptr;
};

} // namespace

class Preview::Impl
{
public:
    Impl() : worker_([this] { run(); })
    {
    }
    ~Impl()
    {
        stop_.store(true);
        wake_.notify_all();
        if (worker_.joinable())
            worker_.join();
    }

    void submit(const CapturedFrame& frame) noexcept
    {
        try
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_.load())
                return;
            if (frame.metadata.pixel_format == core::EgoPixelFormat_Mjpg)
                latest_mjpeg_[frame.metadata.stream] = frame;
            else
            {
                if (encoded_.size() >= kMaxQueuedAccessUnits || frame.encoded_data.size() > kMaxQueuedBytes ||
                    queued_bytes_ > kMaxQueuedBytes - frame.encoded_data.size())
                {
                    lock.unlock();
                    disable("encoded queue exceeded 64 access units or 64 MiB");
                    return;
                }
                encoded_.push_back(frame);
                queued_bytes_ += frame.encoded_data.size();
            }
            wake_.notify_one();
        }
        catch (const std::exception& error)
        {
            disable(error.what());
        }
    }

    bool closed() const
    {
        return closed_.load();
    }

private:
    void disable(const char* reason) noexcept
    {
        stop_.store(true);
        try
        {
            std::lock_guard<std::mutex> lock(mutex_);
            encoded_.clear();
            latest_mjpeg_.clear();
            queued_bytes_ = 0;
            detail::log_warning() << "Ego preview disabled: " << reason << "; recording continues";
        }
        catch (...)
        {
        }
        wake_.notify_all();
    }

    void run()
    {
        if (SDL_Init(SDL_INIT_VIDEO) != 0)
        {
            closed_.store(true);
            return;
        }
        SDL_Window* window = SDL_CreateWindow(
            "EGO Preview", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 520, SDL_WINDOW_RESIZABLE);
        SDL_Renderer* renderer = window ? SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED) : nullptr;
        if (!window || !renderer)
        {
            if (renderer)
                SDL_DestroyRenderer(renderer);
            if (window)
                SDL_DestroyWindow(window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
            closed_.store(true);
            return;
        }
        std::map<core::EgoCameraStream, std::unique_ptr<Decoder>> decoders;
        std::map<core::EgoCameraStream, uint32_t> epochs;
        std::map<core::EgoCameraStream, RgbFrame> images;
        std::map<core::EgoCameraStream, Texture> textures;
        const auto decode_frame = [&](const CapturedFrame& frame)
        {
            const auto stream = frame.metadata.stream;
            auto it = decoders.find(stream);
            if (it == decoders.end() || epochs.at(stream) != frame.metadata.capture_epoch)
            {
                decoders[stream] = std::make_unique<Decoder>(frame.metadata.pixel_format);
                epochs[stream] = frame.metadata.capture_epoch;
                images.erase(stream);
                it = decoders.find(stream);
            }
            RgbFrame image;
            if (it->second->decode(frame, image))
            {
                image.capture_epoch = frame.metadata.capture_epoch;
                images[stream] = std::move(image);
            }
        };
        try
        {
            while (!stop_.load())
            {
                SDL_Event event;
                while (SDL_PollEvent(&event))
                {
                    if (event.type == SDL_QUIT)
                    {
                        closed_.store(true);
                        stop_.store(true);
                    }
                }
                std::deque<CapturedFrame> encoded;
                std::map<core::EgoCameraStream, CapturedFrame> mjpeg;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_.wait_for(lock, std::chrono::milliseconds(10),
                                   [this] { return stop_.load() || !encoded_.empty() || !latest_mjpeg_.empty(); });
                    encoded.swap(encoded_);
                    mjpeg.swap(latest_mjpeg_);
                    queued_bytes_ = 0;
                }
                // Prediction access units must reach the decoder in order; only decoded images may be stale.
                for (const auto& frame : encoded)
                {
                    if (stop_.load())
                        break;
                    decode_frame(frame);
                }
                for (const auto& [_, frame] : mjpeg)
                {
                    if (stop_.load())
                        break;
                    decode_frame(frame);
                }
                if (stop_.load())
                    break;
                const auto left = images.find(core::EgoCameraStream_ColorLeft);
                const auto right = images.find(core::EgoCameraStream_ColorRight);
                if (left != images.end() && right != images.end() &&
                    (left->second.capture_epoch != right->second.capture_epoch ||
                     std::llabs(left->second.device_timestamp_ns - right->second.device_timestamp_ns) > 5'000'000))
                    continue;

                SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                SDL_RenderClear(renderer);
                int window_width = 0;
                int window_height = 0;
                SDL_GetRendererOutputSize(renderer, &window_width, &window_height);
                int index = 0;
                for (const auto& [stream, image] : images)
                {
                    auto& texture = textures[stream];
                    texture.update(renderer, image);
                    if (texture.handle)
                    {
                        const int count = static_cast<int>(images.size());
                        SDL_Rect target{ index * window_width / count, 0, window_width / count, window_height };
                        SDL_RenderCopy(renderer, texture.handle, nullptr, &target);
                    }
                    ++index;
                }
                SDL_RenderPresent(renderer);
            }
        }
        catch (const std::exception& error)
        {
            disable(error.what());
        }
        // Destroy owned textures before SDL_DestroyRenderer frees its associated textures.
        textures.clear();
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }

    std::atomic<bool> stop_{ false };
    std::atomic<bool> closed_{ false };
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    // The worker owns at most one additional bounded batch while decoding.
    static constexpr size_t kMaxQueuedAccessUnits = 64;
    static constexpr size_t kMaxQueuedBytes = 64 * 1024 * 1024;
    std::deque<CapturedFrame> encoded_;
    size_t queued_bytes_ = 0;
    std::map<core::EgoCameraStream, CapturedFrame> latest_mjpeg_;
    std::thread worker_;
};

Preview::Preview() : impl_(std::make_unique<Impl>())
{
}
Preview::~Preview() = default;
void Preview::submit(const CapturedFrame& frame) noexcept
{
    impl_->submit(frame);
}
bool Preview::closed() const
{
    return impl_->closed();
}

} // namespace plugins::ego
