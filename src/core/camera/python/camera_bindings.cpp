// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera/camera_reader.hpp>
#include <camera/cuda_pool.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace core::camera;

namespace
{

const char* name_of(PixelFormat format)
{
    switch (format)
    {
    case PixelFormat::BGRA8:
        return "bgra8";
    case PixelFormat::RGBA8:
        return "rgba8";
    case PixelFormat::NV12:
        return "nv12";
    case PixelFormat::GRAY8:
        return "gray8";
    }
    return "unknown";
}

DeliveryOrder order_of(const std::string& name)
{
    if (name == "newest")
    {
        return DeliveryOrder::NEWEST;
    }
    if (name == "in_order")
    {
        return DeliveryOrder::IN_ORDER;
    }
    throw std::invalid_argument("order is 'newest' or 'in_order', not '" + name + "'");
}

PluginLocation location_of(const std::string& endpoint, const std::string& instance)
{
    PluginLocation plugin;
    plugin.endpoint = endpoint;
    plugin.instance = instance;
    return plugin;
}

// A frame set as Python holds it: by the reader's FrameSet object and by every array that views
// one of its images. It goes back to the plugin when the last of them is gone.
using Lease = std::shared_ptr<FrameSet>;

// The bytes of one plane or of a whole image, with the shape an array of them has.
struct Layout
{
    std::span<const std::byte> bytes;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
};

// The shape an array of one plane of a pixel image has; `bytes` is left for the caller.
Layout plane_layout_of(const StreamDescription& stream, std::uint32_t index)
{
    const PlaneSize size = plane_size_of(stream.pixel_format, stream.width, stream.height, index);
    const auto rows = static_cast<py::ssize_t>(size.rows);
    const auto stride = static_cast<py::ssize_t>(stream.planes[index].stride_bytes);
    const auto row_bytes = static_cast<py::ssize_t>(size.row_bytes);
    switch (stream.pixel_format)
    {
    case PixelFormat::BGRA8:
    case PixelFormat::RGBA8:
        return { {}, { rows, row_bytes / 4, 4 }, { stride, 4, 1 } };
    case PixelFormat::NV12:
        if (index == 1)
        {
            return { {}, { rows, row_bytes / 2, 2 }, { stride, 2, 1 } };
        }
        [[fallthrough]];
    case PixelFormat::GRAY8:
        return { {}, { rows, row_bytes }, { stride, 1 } };
    }
    throw std::invalid_argument("the image has a pixel format this binding does not know");
}

// A read-only array over image memory. Its base holds the frame set, so what the array shows
// stays that frame for as long as the array exists.
py::array array_of(const Layout& layout, const Lease& lease)
{
    auto* owner = new Lease(lease);
    const py::capsule base(owner, [](void* held) { delete static_cast<Lease*>(held); });
    py::array array(py::dtype::of<std::uint8_t>(), layout.shape, layout.strides, layout.bytes.data(), base);
    py::detail::array_proxy(array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return array;
}

// One plane of an image in CUDA memory, as CuPy, PyTorch and Numba take it: an object with
// __cuda_array_interface__. It holds the frame set, and an array made from it holds this.
struct CudaArray
{
    CudaAddress address = 0;
    std::uint32_t device = 0;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    Lease lease;

    py::dict interface() const
    {
        py::dict description;
        description["shape"] = py::tuple(py::cast(shape));
        description["typestr"] = "|u1";
        // The plugin's buffer is mapped read-only on the device. It is described as writable
        // because PyTorch takes nothing else; a write through it is refused by the device.
        description["data"] = py::make_tuple(address, false);
        description["strides"] = py::tuple(py::cast(strides));
        description["version"] = 3;
        return description;
    }
};

// One image of a frame set. It does not hold the frame set itself: its arrays do.
struct Image
{
    ImageView view;
    std::shared_ptr<const GroupDescription> group; // owns the stream description
    std::size_t stream = 0;
    std::weak_ptr<FrameSet> frame_set;

    Lease lease() const
    {
        Lease held = frame_set.lock();
        if (!held)
        {
            throw std::runtime_error("the frame set was given back");
        }
        return held;
    }

    const StreamDescription& description() const
    {
        return group->streams[stream];
    }

    // An image is asked for where it is: in host memory or on a GPU.
    void require_on_gpu(bool on_gpu) const
    {
        if (description().on_gpu() != on_gpu)
        {
            throw std::invalid_argument(on_gpu ? "the image is in host memory: array(), plane(i) and data give it" :
                                                 "the image is on a GPU: cuda_array() and cuda_plane(i) give it");
        }
    }

    // The image with its description where this object keeps it: the reader may be gone.
    ImageView described() const
    {
        ImageView local = view;
        local.stream = &description();
        return local;
    }

    Layout data_layout() const
    {
        return { view.data, { static_cast<py::ssize_t>(view.data.size()) }, { 1 } };
    }

    Layout plane_layout(std::uint32_t index) const
    {
        const std::span<const std::byte> bytes = described().plane(index);
        Layout layout = plane_layout_of(description(), index);
        layout.bytes = bytes;
        return layout;
    }

    py::array data() const
    {
        require_on_gpu(false);
        return array_of(data_layout(), lease());
    }

    py::array plane(std::uint32_t index) const
    {
        require_on_gpu(false);
        const Lease held = lease(); // before the layout, which reads the image's memory
        return array_of(plane_layout(index), held);
    }

    py::array array() const
    {
        require_one_plane("array()", "plane(i)");
        return plane(0);
    }

    CudaArray cuda_plane(std::uint32_t index) const
    {
        require_on_gpu(true);
        Lease held = lease();
        const CudaAddress address = described().cuda_plane(index);
        Layout layout = plane_layout_of(description(), index);
        return { address, view.cuda_device_index, std::move(layout.shape), std::move(layout.strides), std::move(held) };
    }

    CudaArray cuda_array() const
    {
        require_one_plane("cuda_array()", "cuda_plane(i)");
        return cuda_plane(0);
    }

private:
    void require_one_plane(const char* asked, const char* per_plane) const
    {
        if (description().plane_count != 1)
        {
            throw std::invalid_argument(std::string(asked) + " is for images with one plane; use " + per_plane +
                                        " for " + name_of(description().pixel_format));
        }
    }
};

// A frame set as the reader handed it out. What is known about it stays readable after it was
// given back; its images do not.
struct Frames
{
    Frames(FrameSet&& set, std::shared_ptr<const GroupDescription> description)
        : sequence(set.sequence()),
          generation(set.generation()),
          skipped(set.skipped()),
          timeline(set.timeline()),
          image_count(set.image_count()),
          group(std::move(description)),
          lease(std::make_shared<FrameSet>(std::move(set)))
    {
    }

    Image image(std::size_t stream) const
    {
        if (!lease)
        {
            throw std::runtime_error("the frame set was given back");
        }
        return { lease->image(stream), group, stream, lease };
    }

    // Lets go of the frame set. It goes back to the plugin now, or when the last array that
    // views one of its images is gone.
    void release()
    {
        lease.reset();
    }

    // The end of a `with frames.borrow()` block: gives the frame set back, and says so when an
    // array of it still exists instead of leaving the frame set held without a word. When the
    // block ends in an exception the arrays are left to keep it; that error is the one to see.
    void end_borrow(bool failed)
    {
        const bool kept = lease && lease.use_count() > 1;
        lease.reset();
        if (kept && !failed)
        {
            throw py::buffer_error(
                "an array of this frame set still exists at the end of the block it was "
                "borrowed for: drop it before the block ends, or copy what is needed from it");
        }
    }

    std::uint64_t sequence;
    std::uint32_t generation;
    std::uint32_t skipped;
    FrameTimeline timeline;
    std::size_t image_count;
    std::shared_ptr<const GroupDescription> group;
    Lease lease;
};

// `with frames.borrow() as frames`: the frame set for the length of a block.
struct Borrowed
{
    std::shared_ptr<Frames> frames;
};

// How long one wait for a frame set lasts before the interpreter gets to look at signals.
constexpr std::chrono::milliseconds SIGNAL_CHECK_INTERVAL(100);

class Reader
{
public:
    explicit Reader(CameraReader reader)
        : m_group(std::make_shared<const GroupDescription>(reader.group())),
          m_reader(std::make_shared<CameraReader>(std::move(reader)))
    {
    }

    const GroupDescription& group() const
    {
        return *m_group;
    }

    // Waits up to `timeout_seconds`; None when no frame set came.
    std::shared_ptr<Frames> next(double timeout_seconds)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
        for (;;)
        {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            // The wait holds the reader itself: close() on another thread, while this one waits
            // without the interpreter lock, ends the reading and does not take the reader away
            // from under the wait.
            const std::shared_ptr<CameraReader> reader = open_reader();
            std::optional<FrameSet> frames;
            {
                const py::gil_scoped_release release;
                frames = reader->next(std::clamp(left, std::chrono::milliseconds(0), SIGNAL_CHECK_INTERVAL));
            }
            if (m_reader != reader)
            {
                frames.reset(); // closed meanwhile: what arrived goes back
                open_reader();
            }
            if (frames)
            {
                return std::make_shared<Frames>(std::move(*frames), m_group);
            }
            if (left <= std::chrono::milliseconds(0))
            {
                return nullptr;
            }
            if (PyErr_CheckSignals() != 0)
            {
                throw py::error_already_set();
            }
        }
    }

    // For `for frames in reader`: lets go of the frame set handed out before, then waits for
    // the next for as long as it takes.
    std::shared_ptr<Frames> iterate()
    {
        if (m_iterated)
        {
            m_iterated->release();
            m_iterated.reset();
        }
        while (!m_iterated)
        {
            m_iterated = next(std::chrono::duration<double>(SIGNAL_CHECK_INTERVAL).count());
        }
        return m_iterated;
    }

    void close()
    {
        if (m_iterated)
        {
            m_iterated->release();
            m_iterated.reset();
        }
        m_reader.reset();
    }

    int fileno() const
    {
        return open_reader()->descriptor();
    }

private:
    // Call with the interpreter lock held.
    std::shared_ptr<CameraReader> open_reader() const
    {
        if (!m_reader)
        {
            throw CameraReaderError("the camera reader is closed");
        }
        return m_reader;
    }

    std::shared_ptr<const GroupDescription> m_group;
    std::shared_ptr<CameraReader> m_reader; // empty once closed
    std::shared_ptr<Frames> m_iterated;
};

} // namespace

PYBIND11_MODULE(_camera, m)
{
    m.doc() = "Isaac Capture camera reader: frames from a camera plugin, without a copy";

    static py::exception<CameraReaderError> reader_error(m, "CameraReaderError", PyExc_RuntimeError);
    static py::exception<CameraDisconnected> disconnected(m, "CameraDisconnected", reader_error.ptr());
    static py::exception<CameraWindowExhausted> exhausted(m, "CameraWindowExhausted", PyExc_RuntimeError);
    py::register_exception_translator(
        [](std::exception_ptr pointer)
        {
            try
            {
                if (pointer)
                {
                    std::rethrow_exception(pointer);
                }
            }
            catch (const CameraWindowExhausted& error)
            {
                // In Python it is usually an array, not a frame set, that still holds one.
                const std::string message = std::string(error.what()) +
                                            ". An array taken from a frame set keeps it held: drop the array, or "
                                            "copy what is needed from it";
                PyErr_SetString(exhausted.ptr(), message.c_str());
            }
            catch (const CameraDisconnected& error)
            {
                PyErr_SetString(disconnected.ptr(), error.what());
            }
            catch (const CameraReaderError& error)
            {
                PyErr_SetString(reader_error.ptr(), error.what());
            }
        });

    py::class_<StreamDescription>(m, "StreamDescription", "One sensor's images.")
        .def_readonly("id", &StreamDescription::id)
        .def_property_readonly("pixel_format", [](const StreamDescription& s) { return name_of(s.pixel_format); })
        .def_readonly("width", &StreamDescription::width)
        .def_readonly("height", &StreamDescription::height)
        .def_readonly("plane_count", &StreamDescription::plane_count)
        .def_property_readonly(
            "strides",
            [](const StreamDescription& s)
            {
                std::vector<std::uint32_t> strides;
                for (std::uint32_t i = 0; i < s.plane_count; ++i)
                {
                    strides.push_back(s.planes[i].stride_bytes);
                }
                return strides;
            },
            "Bytes from one row to the next, per plane")
        .def_readonly("slot_bytes", &StreamDescription::slot_bytes)
        .def_readonly("slot_count", &StreamDescription::slot_count)
        .def_readonly("frame_rate", &StreamDescription::frame_rate)
        .def_property_readonly(
            "memory", [](const StreamDescription& s) { return s.memory == MemoryKind::CUDA ? "cuda" : "host"; },
            "The kind of memory the images are in: 'host', or 'cuda' for memory of one CUDA device. An "
            "application asks on_gpu, which holds for every kind of GPU memory")
        .def_property_readonly("on_gpu", &StreamDescription::on_gpu,
                               "Whether the images are on a GPU: cuda_array() and cuda_plane(i) then give them, "
                               "and array(), plane(i) and data do not")
        .def_property_readonly(
            "cuda_device",
            [](const StreamDescription& s) -> std::optional<std::string>
            {
                if (s.memory != MemoryKind::CUDA)
                {
                    return std::nullopt;
                }
                return cuda_device_name(s.cuda_device);
            },
            "The device of 'cuda' memory, named as nvidia-smi names it ('GPU-...'); None for 'host' memory")
        .def("__repr__",
             [](const StreamDescription& s)
             {
                 return "<StreamDescription '" + s.id + "' " + std::to_string(s.width) + "x" +
                        std::to_string(s.height) + " " + name_of(s.pixel_format) + ">";
             });

    py::class_<GroupDescription>(
        m, "GroupDescription", "Streams that are captured together and published as one frame set.")
        .def_readonly("id", &GroupDescription::id)
        .def_readonly("session_id", &GroupDescription::session_id,
                      "Names one run of the group; with a frame set's sequence it identifies a capture")
        .def_readonly("streams", &GroupDescription::streams)
        .def("__repr__", [](const GroupDescription& g)
             { return "<GroupDescription '" + g.id + "' with " + std::to_string(g.streams.size()) + " stream(s)>"; });

    py::class_<GroupListing>(m, "GroupListing", "One group a plugin publishes, as CameraReader.list() reports it.")
        .def_property_readonly("id", [](const GroupListing& g) { return g.group.id; })
        .def_property_readonly("streams", [](const GroupListing& g) { return g.group.streams; })
        .def_readonly("unsupported", &GroupListing::unsupported,
                      "Why this build cannot read the group; empty for one it can open")
        .def_readonly("endpoint", &GroupListing::endpoint, "The socket path of the plugin that publishes the group")
        .def("__repr__",
             [](const GroupListing& g)
             {
                 return "<GroupListing '" + g.group.id + "'" +
                        (g.unsupported.empty() ? " with " + std::to_string(g.group.streams.size()) + " stream(s)" :
                                                 ", not readable by this build") +
                        (g.endpoint.empty() ? "" : " on '" + g.endpoint + "'") + ">";
             });

    py::class_<FrameTimeline>(m, "FrameTimeline",
                              "Where a frame set was at each step: nanoseconds on the host's monotonic clock "
                              "(time.monotonic_ns()), -1 where a time is not known.")
        .def_readonly("capture_ns", &FrameTimeline::capture_ns)
        .def_readonly("sdk_ready_ns", &FrameTimeline::sdk_ready_ns)
        .def_readonly("published_ns", &FrameTimeline::published_ns)
        .def_readonly("sent_ns", &FrameTimeline::sent_ns);

    py::class_<CudaArray>(
        m, "CudaArray",
        "One plane of an image in CUDA memory: the plugin's buffer on the device, not a copy. CuPy, PyTorch "
        "and Numba make an array of it without a copy, through __cuda_array_interface__: "
        "cupy.asarray(plane), torch.as_tensor(plane, device=f'cuda:{plane.device}'). It keeps its frame set "
        "held, and so does every array made from it. The buffer is read-only: the device refuses a write to "
        "it. Keep the array until the device has finished what reads it; when the frame set goes back, the "
        "camera writes the buffer again.")
        .def_property_readonly("__cuda_array_interface__", &CudaArray::interface)
        .def_property_readonly(
            "shape", [](const CudaArray& a) { return py::tuple(py::cast(a.shape)); },
            "As ImageView.array() and plane(): (height, width, 4), (height, width) or (height/2, width/2, 2)")
        .def_readonly("device", &CudaArray::device, "The index, in this process, of the device the buffer is on")
        .def("__repr__",
             [](const CudaArray& a)
             {
                 return "<CudaArray " + std::string(py::str(py::tuple(py::cast(a.shape)))) +
                        " on cuda:" + std::to_string(a.device) + ">";
             });

    py::class_<Image>(m, "ImageView",
                      "One image of a frame set. Its arrays are read-only views of the plugin's buffer: no copy "
                      "is made. An array keeps its frame set held, so it shows that frame for as long as it "
                      "exists and counts against the reader's window until it is gone. An image is in the "
                      "host memory or on a GPU, as stream.on_gpu says: array(), plane(i) and data are for "
                      "host memory, cuda_array() and cuda_plane(i) for an image on a GPU.")
        .def_property_readonly("stream", &Image::description)
        .def_property_readonly("slot", [](const Image& i) { return i.view.info.slot; })
        .def_property_readonly("flags", [](const Image& i) { return i.view.info.flags; })
        .def_property_readonly("device_frame_id", [](const Image& i) { return i.view.info.device_frame_id; })
        .def_property_readonly("device_time_ns", [](const Image& i) { return i.view.info.device_time_ns; })
        .def_property_readonly("data", &Image::data, "The image's bytes as a flat array: the whole slot")
        .def("array", &Image::array,
             "The pixels as (height, width, 4) for bgra8 and rgba8, (height, width) for gray8. Rows are stride bytes apart.")
        .def("plane", &Image::plane, py::arg("index"),
             "One plane: nv12 has (height, width) luma and (height/2, width/2, 2) chroma.")
        .def("cuda_array", &Image::cuda_array, "array() for an image on a GPU: the pixels on the device")
        .def("cuda_plane", &Image::cuda_plane, py::arg("index"),
             "plane() for an image on a GPU: one plane on the device");

    py::class_<Borrowed>(m, "_Borrowed")
        .def("__enter__", [](const Borrowed& borrowed) { return borrowed.frames; })
        .def("__exit__", [](Borrowed& borrowed, const py::object& error, const py::object&, const py::object&)
             { borrowed.frames->end_borrow(!error.is_none()); });

    py::class_<Frames, std::shared_ptr<Frames>>(
        m, "FrameSet",
        "One frame set, held for this reader: the plugin does not reuse its buffers until it is given back. "
        "release(), leaving a `with` block or the object going away let go of it; it goes back then, or when "
        "the last array that views one of its images is gone.")
        .def_readonly("sequence", &Frames::sequence,
                      "The frame set's place among the group's captures; the same for every reader")
        .def_readonly("generation", &Frames::generation, "Restarts of the camera since the plugin began publishing")
        .def_readonly("skipped", &Frames::skipped,
                      "Frame sets published since the previous one this reader was given, and not given to it")
        .def_readonly("timeline", &Frames::timeline)
        .def("__len__", [](const Frames& f) { return f.image_count; })
        .def("image", &Frames::image, py::arg("stream"), "The image of one stream, in the order of the group's streams")
        .def("release", &Frames::release, "Let go of the frame set; arrays of its images keep it until they are gone")
        .def(
            "borrow", [](const std::shared_ptr<Frames>& f) { return Borrowed{ f }; },
            "For `with frames.borrow() as frames`: the frame set goes back to the plugin when the block ends, and "
            "every array taken from it has to be gone by then. One that still exists raises BufferError.")
        .def("__enter__", [](const std::shared_ptr<Frames>& f) { return f; })
        .def("__exit__", [](Frames& f, const py::object&, const py::object&, const py::object&) { f.release(); });

    py::class_<Reader>(m, "CameraReader", "Reads one group of cameras from the camera plugin that publishes it.")
        .def_static(
            "open",
            [](const std::string& group, std::uint32_t window, const std::string& order, const std::string& endpoint,
               const std::string& instance)
            {
                const ReaderOptions options{ window, order_of(order) };
                const PluginLocation plugin = location_of(endpoint, instance);
                const py::gil_scoped_release release;
                return std::make_unique<Reader>(CameraReader::open(group, options, plugin));
            },
            py::arg("group"), py::kw_only(), py::arg("window") = 2, py::arg("order") = "newest",
            py::arg("endpoint") = "", py::arg("instance") = "",
            "Subscribe to one group. `window` is how many frame sets this reader may hold at a time; the default "
            "of two lets a loop keep the array of one frame while it asks for the next. `order` is 'newest' (each "
            "call gets the latest) or 'in_order' (every frame set). The reader looks through this user's camera "
            "plugins for the one that publishes the group; `instance` or `endpoint` names one plugin instead, by "
            "its instance name or its socket path. Raises CameraReaderError, which says why, when no plugin publishes "
            "the group or two do, the plugin that was named does not answer, or the plugin refuses the reader.")
        .def_static(
            "list",
            [](const std::string& endpoint, const std::string& instance)
            {
                const PluginLocation plugin = location_of(endpoint, instance);
                const py::gil_scoped_release release;
                return CameraReader::list(plugin);
            },
            py::kw_only(), py::arg("endpoint") = "", py::arg("instance") = "",
            "What is published, each group with the `endpoint` of its plugin: by every camera plugin of this user "
            "that says what it publishes, or by the one plugin `instance` or `endpoint` names. No plugin running "
            "gives an empty list; a plugin that was named and does not answer raises CameraReaderError.")
        .def_property_readonly("group", &Reader::group)
        .def(
            "next",
            // An optional, so that the signature and the generated stub say that None can come back.
            [](Reader& reader, double timeout) -> std::optional<std::shared_ptr<Frames>>
            {
                std::shared_ptr<Frames> frames = reader.next(timeout);
                if (!frames)
                {
                    return std::nullopt;
                }
                return frames;
            },
            py::arg("timeout") = 1.0,
            "Wait up to `timeout` seconds for a frame set; None when none came. Raises CameraWindowExhausted "
            "when the reader already holds as many as its window.")
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &Reader::iterate,
             "The next frame set, waiting as long as it takes. The one handed out before is let go of first.")
        .def("close", &Reader::close,
             "Stop reading. Frame sets and arrays that still exist stay as they are, and the connection to the "
             "plugin ends when the last of them is gone. May be called from another thread than the one that "
             "reads: a next() that is waiting then raises CameraReaderError within a tenth of a second.")
        .def("fileno", &Reader::fileno,
             "A descriptor that is readable when a frame set may be waiting, for select and asyncio. In newest "
             "order the plugin sends a frame set only when it is asked: call next(0) once to ask, and again each "
             "time the descriptor is readable, until it returns None.")
        .def("__enter__", [](py::object self) { return self; })
        .def("__exit__", [](Reader& reader, const py::object&, const py::object&, const py::object&) { reader.close(); });
}
