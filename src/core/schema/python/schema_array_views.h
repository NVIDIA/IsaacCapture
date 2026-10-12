// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Helpers for exposing FlatBuffers struct arrays and vectors to Python as NumPy views.
//
// A FlatBuffers struct array is interleaved (array of structs), so a field is exposed as a
// strided view rather than a packed buffer; NumPy and DLPack both carry strides natively,
// so such views satisfy the retargeting engine's NDArrayType tensor contract without a copy.
//
// Two ways to reach a field, in order of preference:
//
//   1. Take its address through the generated accessor. flatc returns nested structs by
//      const reference, so &joint.pose().position() is an ordinary member address: the
//      compiler does the arithmetic, and renaming or removing the field in the .fbs breaks
//      the build. No offset is involved.
//   2. Only when 1 is impossible: FBS_FIELD_OFFSET below. flatc returns scalars *by value*
//      (through EndianScalar), so scalar fields have no addressable accessor and their
//      offset has to be looked up.

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

// These views hand NumPy a raw pointer into FlatBuffers storage, so they bypass the
// EndianScalar() conversion the generated accessors apply. FlatBuffers stores scalars
// little-endian and NumPy reads them in host order, so the two agree only on a
// little-endian host — as do all supported targets (x86_64, ARM64), where EndianScalar()
// compiles to a no-op. Fail the build rather than silently byte-swapped data if that ever
// stops being true; a big-endian port would need packed copies, not views.
static_assert(FLATBUFFERS_LITTLEENDIAN == 1,
              "Strided FlatBuffers views assume a little-endian host; on big-endian the "
              "generated accessors byte-swap and these raw-pointer views would not");

// Byte offset of a struct field. Call it through FBS_FIELD_OFFSET rather than directly — the
// accessor argument is what makes the field name compile-time checked.
//
// The offset comes from flatc's reflection table, emitted under --reflect-names (see
// cmake/GenerateFlatBuffers.cmake), so reordering or inserting schema fields cannot shift
// what gets read. A fully compile-time offset is not reachable for these types:
// MiniReflectTypeTable() is not constexpr, and the generated members are private so offsetof
// cannot name them. Hence a runtime lookup, resolved once at binding time, that throws on
// miss rather than returning a sentinel.
template <typename StructT, typename Accessor>
inline py::ssize_t fbs_field_offset(const char* field_name, Accessor /*compile-time probe*/)
{
    const ::flatbuffers::TypeTable* type_table = StructT::MiniReflectTypeTable();
    if (type_table->names == nullptr)
    {
        throw std::runtime_error(
            "FlatBuffers headers were generated without --reflect-names; "
            "schema field offsets cannot be resolved by name");
    }
    // values holds byte offsets only for structs; tables leave it null (their fields live behind
    // a vtable and have no fixed offset), so reject those before indexing it.
    if (type_table->st != ::flatbuffers::ST_STRUCT || type_table->values == nullptr)
    {
        throw std::runtime_error(
            std::string("field offsets are only defined for FlatBuffers structs, not for the type owning: ") + field_name);
    }
    for (size_t i = 0; i < type_table->num_elems; ++i)
    {
        if (std::strcmp(type_table->names[i], field_name) == 0)
        {
            return static_cast<py::ssize_t>(type_table->values[i]);
        }
    }
    throw std::runtime_error(std::string("no such field in FlatBuffers struct: ") + field_name);
}

// Byte offset of `field` within FlatBuffers struct `StructT`, naming the field exactly once.
//
// The name reaches fbs_field_offset() two ways: stringified for the reflection lookup, and
// as &StructT::field. The second is a compile-time probe with no runtime effect — rename or
// remove the field in the .fbs and the generated accessor goes with it, so this stops
// COMPILING instead of throwing at import or, worse, reading the wrong bytes.
#define FBS_FIELD_OFFSET(StructT, field) (::core::fbs_field_offset<StructT>(#field, &StructT::field))

// Address of a scalar field that has no addressable accessor, given its offset. Only needed
// for case 2 above; prefer taking the accessor's address directly.
template <typename T, typename StructT>
inline const T* fbs_field_address(const StructT& value, py::ssize_t field_offset)
{
    return reinterpret_cast<const T*>(reinterpret_cast<const std::byte*>(&value) + field_offset);
}

// Strided view of one field across an array of `count` structs, starting at `first` and
// repeating every `stride` bytes. `width` is the trailing dimension (0 for a scalar field).
// `owner` becomes the array's base, so the storage it aliases stays alive.
//
// The view is writable, and writes go straight back into the schema object. Marking it
// read-only would be the safer contract, but NumPy cannot export a read-only array over
// DLPack before 2.1 (BufferError), and the retargeting engine reaches every tensor through
// np.from_dlpack — so a read-only view would break the wheel's declared numpy>=1.23 floor.
// Callers that intend to modify the data must .copy() first.
template <typename T>
inline py::array_t<T> strided_field_view(
    py::object owner, const T* first, py::ssize_t stride, py::ssize_t count, py::ssize_t width)
{
    if (width > 0)
    {
        return py::array_t<T>({ count, width }, { stride, static_cast<py::ssize_t>(sizeof(T)) }, first, owner);
    }
    return py::array_t<T>({ count }, { stride }, first, owner);
}

// (count, Width) view over a FlatBuffers vector of structs that are exactly Width packed T
// (Point as 3 floats, say). FlatBuffers omits an empty vector, so absent reads as (0, Width).
template <typename T, py::ssize_t Width, typename StructT>
inline py::array_t<T> struct_vector_view(py::object owner, const ::flatbuffers::Vector<const StructT*>* values)
{
    static_assert(sizeof(StructT) == sizeof(T) * Width, "struct must be exactly Width packed elements");
    if (values == nullptr || values->size() == 0)
    {
        return py::array_t<T>(std::vector<py::ssize_t>{ 0, Width });
    }
    return strided_field_view<T>(owner, reinterpret_cast<const T*>(values->Data()),
                                 static_cast<py::ssize_t>(sizeof(StructT)), values->size(), Width);
}

// (count,) view over a FlatBuffers vector of scalars; absent reads as (0,).
template <typename T>
inline py::array_t<T> scalar_vector_view(py::object owner, const ::flatbuffers::Vector<T>* values)
{
    if (values == nullptr || values->size() == 0)
    {
        return py::array_t<T>(py::ssize_t{ 0 });
    }
    return strided_field_view<T>(owner, values->data(), static_cast<py::ssize_t>(sizeof(T)), values->size(), 0);
}

// Rows of a (count, Width) array as structs that are exactly Width packed T, for a table
// constructor. Raises ValueError naming `field` on any other shape.
template <typename StructT, typename T, py::ssize_t Width>
inline std::vector<StructT> structs_from_rows(const py::array_t<T, py::array::c_style | py::array::forcecast>& rows,
                                              const char* field)
{
    static_assert(sizeof(StructT) == sizeof(T) * Width, "struct must be exactly Width packed elements");
    if (rows.ndim() != 2 || rows.shape(1) != Width)
    {
        throw py::value_error(std::string(field) + ": expected an (N, " + std::to_string(Width) + ") array");
    }
    std::vector<StructT> out(static_cast<size_t>(rows.shape(0)));
    if (!out.empty())
    {
        std::memcpy(static_cast<void*>(out.data()), rows.data(), out.size() * sizeof(StructT));
    }
    return out;
}

} // namespace core
