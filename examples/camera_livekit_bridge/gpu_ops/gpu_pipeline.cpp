// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/gpu_ops/gpu_step.hpp"
#include "inc/gpu_ops/sine_warp.hpp"

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace camera_bridge
{

namespace
{

/// A plain copy: what the pipeline does when it has no steps.
cudaError_t copy_nv12(const Nv12Planes& in, const Nv12PlanesOut& out, const ImageSize& size, cudaStream_t stream)
{
    const cudaError_t y = cudaMemcpy2DAsync(
        out.y, out.y_pitch, in.y, in.y_pitch, size.width, size.height, cudaMemcpyDeviceToDevice, stream);
    if (y != cudaSuccess)
        return y;
    return cudaMemcpy2DAsync(
        out.uv, out.uv_pitch, in.uv, in.uv_pitch, size.width, size.height / 2, cudaMemcpyDeviceToDevice, stream);
}

class BoxDownscaleStep final : public GpuStep
{
public:
    explicit BoxDownscaleStep(uint32_t divisor) : m_divisor(divisor)
    {
    }

    const char* name() const override
    {
        return "box-downscale";
    }

    ImageSize output_size(const ImageSize& input) const override
    {
        return { input.width / m_divisor, input.height / m_divisor };
    }

    std::string prepare(const ImageSize& input) override
    {
        m_size = input;
        return box_downscale_supported(input.width, input.height, m_divisor) ?
                   std::string() :
                   "a " + std::to_string(input.width) + "x" + std::to_string(input.height) +
                       " picture cannot be shrunk by " + std::to_string(m_divisor) + " to an even size";
    }

    cudaError_t enqueue(const Nv12Planes& in, const Nv12PlanesOut& out, cudaStream_t stream) override
    {
        return box_downscale_nv12(in, m_size.width, m_size.height, out, m_divisor, stream);
    }

private:
    uint32_t m_divisor;
    ImageSize m_size;
};

std::vector<float> parse_floats(const std::string& text)
{
    std::vector<float> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        char* end = nullptr;
        const float value = std::strtof(item.c_str(), &end);
        if (item.empty() || end == nullptr || *end != '\0')
            throw std::invalid_argument("not a number: '" + item + "'");
        values.push_back(value);
    }
    return values;
}

} // namespace

GpuPipeline::~GpuPipeline()
{
    for (auto& scratch : m_scratch)
    {
        cudaFree(scratch.y);
        cudaFree(scratch.uv);
    }
}

void GpuPipeline::add(std::unique_ptr<GpuStep> step)
{
    m_steps.push_back(std::move(step));
}

std::string GpuPipeline::prepare(const ImageSize& source)
{
    m_source = source;
    m_inputs.clear();
    ImageSize size = source;
    for (const auto& step : m_steps)
    {
        m_inputs.push_back(size);
        const std::string reason = step->prepare(size);
        if (!reason.empty())
            return std::string(step->name()) + ": " + reason;
        size = step->output_size(size);
        if (size.width == 0 || size.height == 0 || size.width % 2 != 0 || size.height % 2 != 0)
            return std::string(step->name()) + ": it would produce a " + std::to_string(size.width) + "x" +
                   std::to_string(size.height) + " picture, and NV12 needs even sizes";
    }
    m_output = size;
    // One buffer for the output of every step except the last, which writes into the caller's destination.
    for (std::size_t i = 0; i + 1 < m_steps.size(); ++i)
    {
        const ImageSize next = m_steps[i]->output_size(m_inputs[i]);
        Scratch scratch;
        if (cudaMallocPitch(reinterpret_cast<void**>(&scratch.y), &scratch.y_pitch, next.width, next.height) !=
                cudaSuccess ||
            cudaMallocPitch(reinterpret_cast<void**>(&scratch.uv), &scratch.uv_pitch, next.width, next.height / 2) !=
                cudaSuccess)
        {
            cudaFree(scratch.y);
            return "cannot allocate the buffer between " + std::string(m_steps[i]->name()) + " and the next step";
        }
        m_scratch.push_back(scratch);
    }
    return {};
}

cudaError_t GpuPipeline::enqueue(const Nv12Planes& source, const Nv12PlanesOut& destination, cudaStream_t stream)
{
    if (m_steps.empty())
        return copy_nv12(source, destination, m_source, stream);
    Nv12Planes current = source;
    for (std::size_t i = 0; i < m_steps.size(); ++i)
    {
        const bool last = i + 1 == m_steps.size();
        const Nv12PlanesOut out =
            last ? destination :
                   Nv12PlanesOut{ m_scratch[i].y, m_scratch[i].uv, m_scratch[i].y_pitch, m_scratch[i].uv_pitch };
        const cudaError_t status = m_steps[i]->enqueue(current, out, stream);
        if (status != cudaSuccess)
            return status;
        current = { out.y, out.uv, out.y_pitch, out.uv_pitch };
    }
    return cudaSuccess;
}

std::unique_ptr<GpuStep> make_box_downscale_step(uint32_t divisor)
{
    return std::make_unique<BoxDownscaleStep>(divisor);
}

// The one place that maps a step name to a step. To add a step, add a branch here.
std::unique_ptr<GpuStep> make_gpu_step(const std::string& spec)
{
    const auto equals = spec.find('=');
    const std::string name = spec.substr(0, equals);
    const std::string args = equals == std::string::npos ? std::string() : spec.substr(equals + 1);
    if (name == "sine-warp")
    {
        // sine-warp[=AMPLITUDE_PX[,WAVELENGTH_PX[,PHASE_RAD]]]: an EXAMPLE ripple, not rectification.
        SineWarpParams params;
        const auto values = parse_floats(args);
        if (values.size() > 3)
            throw std::invalid_argument("sine-warp takes at most AMPLITUDE_PX,WAVELENGTH_PX,PHASE_RAD");
        if (values.size() > 0)
            params.amplitude_px = values[0];
        if (values.size() > 1)
            params.wavelength_px = values[1];
        if (values.size() > 2)
            params.phase_rad = values[2];
        return std::make_unique<SineWarpStep>(params);
    }
    // Add steps here, for example:
    //   if (name == "rectify") return std::make_unique<RectifyStep>(load_calibration(args));
    throw std::invalid_argument("unknown GPU step '" + name + "' (known: sine-warp)");
}

} // namespace camera_bridge
