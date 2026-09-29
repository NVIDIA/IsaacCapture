// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io_exo_glove_ros2_plugin.hpp"

#include <string>
#include <vector>

namespace plugins
{
namespace io_exo_glove_ros2
{

//! File name looked up inside a `config/` directory.
inline constexpr const char* kConfigFileName = "io_exo_glove_ros2.yaml";

//! Environment variable naming a config file explicitly, the counterpart of `--config=...`.
inline constexpr const char* kConfigEnvVar = "IO_EXO_GLOVE_ROS2_CONFIG";

//! Where the effective options came from, for startup logging and `--print-config`.
struct ConfigResolution
{
    //! Config file that was loaded; empty when none was found and the built-in defaults were kept.
    std::string loaded_path;
    //! Candidate paths that were probed, in order (diagnostics only).
    std::vector<std::string> searched_paths;
};

//! Overrides the keys a YAML config file defines into @p options; keys the file omits keep their
//! current value, so the caller can layer command-line overrides on top.
//!
//! Search order, first hit wins:
//!   1. @p explicit_path (from `--config=...`)
//!   2. `$IO_EXO_GLOVE_ROS2_CONFIG`
//!   3. `./config/io_exo_glove_ros2.yaml` relative to the working directory -- the plugin launcher
//!      chdir()s into the plugin directory, which is where the file is installed
//!
//! Nothing is baked in at build time: a binary run from a directory without a `config/`
//! subdirectory keeps the built-in defaults unless a file is named explicitly.
//!
//! An explicitly named file (1 or 2) must exist -- it is an error rather than a silent fallback to
//! defaults. Unknown keys and malformed or empty values are rejected too, so a typo never quietly
//! leaves a stale default in place. Throws std::runtime_error on any of those failures.
ConfigResolution load_options_from_config(const std::string& explicit_path, IoExoGloveRos2Plugin::Options& options);

} // namespace io_exo_glove_ros2
} // namespace plugins
