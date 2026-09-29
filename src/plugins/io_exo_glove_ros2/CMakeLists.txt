# SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping io_exo_glove_ros2 plugin (Linux only)")
    add_custom_target(io_exo_glove_ros2_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping io_exo_glove_ros2: Linux only")
    return()
endif()

# rclcpp / sensor_msgs are ordinary ament CMake packages: find_package() locates them once the
# ROS 2 (Humble) environment is sourced (CMAKE_PREFIX_PATH from setup.bash), same as any other
# ROS 2 C++ node -- no colcon/ament build wrapper needed for a plain consumer like this plugin.
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(io_exo_glove_ros2_plugin
    main.cpp
    io_exo_glove_ros2_config.cpp
    io_exo_glove_ros2_plugin.cpp
)

target_link_libraries(io_exo_glove_ros2_plugin PRIVATE
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
    rclcpp::rclcpp
    sensor_msgs::sensor_msgs
    yaml-cpp::yaml-cpp
)

install(TARGETS io_exo_glove_ros2_plugin RUNTIME DESTINATION plugins/io_exo_glove_ros2)
install(FILES plugin.yaml README.md DESTINATION plugins/io_exo_glove_ros2)
install(FILES config/io_exo_glove_ros2.yaml DESTINATION plugins/io_exo_glove_ros2/config)
