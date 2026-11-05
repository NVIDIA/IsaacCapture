# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(revohuman_hand_kinematics STATIC
    hand_geometry.cpp
    hand_geometry.hpp
    openxr_model.inc
    hand_kinematics.cpp
    inc/revohuman_plugin/hand_kinematics.hpp
)
target_include_directories(revohuman_hand_kinematics INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)
target_link_libraries(revohuman_hand_kinematics PUBLIC
    revohuman_plugin::revohuman_core
    revohuman_plugin::sdk_headers)
add_library(revohuman_plugin::revohuman_hand_kinematics ALIAS revohuman_hand_kinematics)

# Optional SDK entry points: an SDK without them still builds, and the plugin stops at startup naming the
# missing one.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES "${REVOHUMAN_SDK_INCLUDE_DIR}")
set(CMAKE_REQUIRED_LIBRARIES "${REVOHUMAN_SDK_LIBRARY}")
# Model-selected point set, the only source of hand poses: without it the plugin fails at startup.
unset(REVOHUMAN_HAS_SDK_POINTS CACHE)
check_cxx_source_compiles([=[
#include <revohuman_fk.h>
int main() {
    double q[RH_JOINT_COUNT]{};
    RhHandModelInfo info = RH_STRUCT_INIT(RhHandModelInfo);
    RhHandPoints points = RH_STRUCT_INIT(RhHandPoints);
    static_assert(RH_HAND_POINT_COUNT == 27);
    return rh_hand_model_info(RH_HAND_MODEL_DV1, &info) + (info.points_version != RH_HAND_POINTS_VERSION)
        + rh_hand_points(RH_HAND_MODEL_P, RH_HAND_LEFT, q, 0, &points);
}
]=] REVOHUMAN_HAS_SDK_POINTS)
# The SDK's product -> hand model table: without it live refuses to start, it never assumes a model.
unset(REVOHUMAN_HAS_SDK_VARIANT_MODEL CACHE)
check_cxx_source_compiles([=[
#include <revohuman_fk.h>
int main() {
    RhHandModel model = 0;
    return rh_variant_hand_model("PTV1", &model) + (model != RH_HAND_MODEL_DV1);
}
]=] REVOHUMAN_HAS_SDK_VARIANT_MODEL)
# Where each taxel sits on the hand, the only source of the TactileMetadata: without it tactile is not pushed.
unset(REVOHUMAN_HAS_SDK_TAXEL_LAYOUT CACHE)
check_cxx_source_compiles([=[
#include <revohuman_fk.h>
static RhTaxelLayout layout = RH_STRUCT_INIT(RhTaxelLayout);
int main() {
    static_assert(RH_TAXEL_LAYOUT_VERSION == 1);
    return rh_hand_model_taxel_layout(RH_HAND_MODEL_DV1, RH_HAND_LEFT, RH_TACTILE_POINTS_TYPE_B, &layout)
        + (layout.taxels[0].link >= RH_JOINT_COUNT);
}
]=] REVOHUMAN_HAS_SDK_TAXEL_LAYOUT)
cmake_pop_check_state()
target_compile_definitions(revohuman_hand_kinematics PRIVATE
    REVOHUMAN_HAS_SDK_POINTS=$<BOOL:${REVOHUMAN_HAS_SDK_POINTS}>
    REVOHUMAN_HAS_SDK_VARIANT_MODEL=$<BOOL:${REVOHUMAN_HAS_SDK_VARIANT_MODEL}>
    REVOHUMAN_HAS_SDK_TAXEL_LAYOUT=$<BOOL:${REVOHUMAN_HAS_SDK_TAXEL_LAYOUT}>)
if(REVOHUMAN_HAS_SDK_POINTS OR REVOHUMAN_HAS_SDK_VARIANT_MODEL OR REVOHUMAN_HAS_SDK_TAXEL_LAYOUT)
    target_link_libraries(revohuman_hand_kinematics PRIVATE revohuman_plugin::sdk)
endif()
