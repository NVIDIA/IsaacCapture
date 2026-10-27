/*
** Copyright (c) 2017-2022, The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <openxr/openxr.h>

// These are just convenience macros for defining enums that would
// normally be declared in the body of the enum, but can't be declared
// that way easily for experimental extensions.

#define XR_ENUM(type, enm, constant) static const type enm = (type)constant
#define XR_STRUCT_ENUM(enm, constant) XR_ENUM(XrStructureType, enm, constant)
#define XR_RESULT_ENUM(enm, constant) XR_ENUM(XrResult, enm, constant)
#define XR_REFSPACE_ENUM(enm, constant) XR_ENUM(XrReferenceSpaceType, enm, constant)
