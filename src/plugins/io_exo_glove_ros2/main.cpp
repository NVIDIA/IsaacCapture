// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_aim.hpp"
#include "io_exo_glove_ros2_config.hpp"
#include "io_exo_glove_ros2_pack.hpp"
#include "io_exo_glove_ros2_plugin.hpp"

#include <log_bridge/logger.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace plugins::io_exo_glove_ros2;

namespace
{

//! This executable's diagnostics go through the same logger as the plugin's; --help and
//! --print-config stay on stdout because they are the program's output, not a log line.
std::shared_ptr<spdlog::logger> logger()
{
    static const std::shared_ptr<spdlog::logger> instance =
        isaaccapture::Logger::get("isaaccapture.plugins.io_exo_glove_ros2.Main");
    return instance;
}

//! Values taken from the command line. A field left unset keeps whatever the config file or the
//! built-in default provided.
struct CliOverrides
{
    std::optional<std::string> joints_topic;
    std::optional<std::string> links_topic;
    std::optional<std::string> joints_left_collection_id;
    std::optional<std::string> joints_right_collection_id;
    std::optional<std::string> tips_left_collection_id;
    std::optional<std::string> tips_right_collection_id;
    std::optional<std::string> tips_reference_frame;
    std::optional<std::string> wrist_source;
    std::optional<std::string> left_aim_to_wrist;
    std::optional<std::string> right_aim_to_wrist;
    std::optional<double> wrist_smoothing_ms;
    //! One-shot calibrations; CLI-only, not config-file keys. The aim offsets are per-run and shared
    //! by both hands, because the controller mounting is the same left and right; the zero mode
    //! prints one config line per hand.
    bool calibrate_aim = false;
    bool calibrate_zero = false;
    bool aim_flip = false;
    std::optional<double> aim_behind_m;
    std::optional<double> aim_height_m;
};

void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options] [--ros-args ...]\n"
              << "\nTopics, collection ids, the fingertip reference frame and the wrist source come from the\n"
              << "config file; the flags below override it.\n"
              << "\nOptions:\n"
              << "  --config=PATH                  Config file to load (default: the default path, or $"
              << kConfigEnvVar << ")\n"
              << "  --joints-topic=TOPIC           Override the glove joint angles topic (sensor_msgs/JointState)\n"
              << "  --links-topic=TOPIC            Override the exoskeleton link poses topic (tf2_msgs/TFMessage)\n"
              << "  --joints-left-collection-id=ID Override the left hand's joint angles collection id\n"
              << "  --joints-right-collection-id=ID\n"
              << "                                 Override the right hand's joint angles collection id\n"
              << "  --tips-left-collection-id=ID   Override the left fingertip poses collection id\n"
              << "  --tips-right-collection-id=ID  Override the right fingertip poses collection id\n"
              << "  --tips-reference-frame=FRAME   Override the frame the fingertip poses are expressed in\n"
              << "  --wrist-source=SOURCE          Override where the injected hand's wrist pose comes from\n"
              << "                                 (auto | hand_tracking | controller)\n"
              << "  --left-aim-to-wrist=POSE       Override the controller aim -> wrist offset, left hand\n"
              << "  --right-aim-to-wrist=POSE      Override it for the right hand\n"
              << "                                 (POSE is x,y,z,qx,qy,qz,qw)\n"
              << "  --wrist-smoothing-ms=MS        Override the tracked wrist's low-pass time constant (0 = off)\n"
              << "  --calibrate-aim                One-shot: hold both hands flat and still, print the controller\n"
              << "                                 aim -> wrist values for both hands and exit\n"
              << "                                 (requires --wrist-source=controller)\n"
              << "  --aim-flip                     Calibration: take the controller aim frame's up axis as\n"
              << "                                 towards the palm, i.e. flip the dorsal hint and roll the\n"
              << "                                 aim -> wrist constant 180 degrees about the fingers\n"
              << "  --aim-behind=M                 Calibration: wrist offset behind the aim origin along the\n"
              << "                                 fingers, in metres (default 0.04)\n"
              << "  --aim-height=M                 Calibration: wrist offset below the aim origin, in metres\n"
              << "                                 (default 0.06)\n"
              << "  --calibrate-zero               One-shot: hold both hands open and straight, print the per-joint\n"
              << "                                 open-hand zero offsets for both hands and exit\n"
              << "                                 (the glove link stream is the input; no controller needed)\n"
              << "  --print-config                 Print the resolved settings and the file they came from, then exit\n"
              << "  --help                         Show this help\n"
              << "\nEverything from --ros-args onwards is handed to ROS 2 unchanged." << std::endl;
}

//! Reports an empty `--key=` value the way an unknown flag is reported, so CLI misuse always prints
//! the usage instead of silently keeping a default.
int fail_empty_flag(const char* prog, const char* flag)
{
    std::cerr << "Error: --" << flag << "= must not be empty." << std::endl;
    print_usage(prog);
    return 1;
}

//! --print-config spelling for the optional per-joint zero offsets: the value, or what an empty one
//! means, so the line documents the key even when the baked table is in use.
std::string zero_offsets_text(const std::string& value)
{
    return value.empty() ? std::string("<empty: baked table>") : value;
}

//! Returns the value of a `--key=value` argument, or nullopt when @p arg is not that key.
std::optional<std::string> flag_value(const std::string& arg, const char* key)
{
    const std::string prefix = std::string("--") + key + "=";
    if (arg.rfind(prefix, 0) != 0)
    {
        return std::nullopt;
    }
    return arg.substr(prefix.size());
}

//! Parses a non-negative length in metres for the calibration flags; prints the usage on bad input,
//! which the one-shot mode cannot recover from.
bool parse_metres(const std::string& value, const char* flag, const char* prog, double& out)
{
    try
    {
        out = std::stod(value);
    }
    catch (const std::exception&)
    {
        std::cerr << "Error: --" << flag << "= must be a number." << std::endl;
        print_usage(prog);
        return false;
    }
    if (!std::isfinite(out) || out < 0.0)
    {
        std::cerr << "Error: --" << flag << "= must be a non-negative number of metres." << std::endl;
        print_usage(prog);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
try
{
    IoExoGloveRos2Plugin::Options options; // Built-in defaults; the config file and CLI layer on top.
    CliOverrides cli;
    std::string config_path;
    bool print_config = false;

    // Arguments handed to ROS: everything this plugin does not consume itself.
    std::vector<std::string> ros_args{ argv[0] };

    bool in_ros_args = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        // Injected by the PluginManager for every plugin it launches, and appended after this
        // plugin's own arguments -- so it can land after --ros-args too. It must never be forwarded
        // to ROS: an unknown token inside the --ros-args section makes rclcpp::init() throw, which
        // would abort startup.
        if (arg.rfind("--plugin-root-id=", 0) == 0)
        {
            continue;
        }

        if (arg == "--ros-args")
        {
            // ROS 2 owns the remainder (remappings, parameter files, logging): forward it untouched
            // from here on -- minus the launcher-owned flag dropped above -- and stop looking for our
            // own flags.
            in_ros_args = true;
            ros_args.push_back(arg);
            continue;
        }
        if (in_ros_args)
        {
            ros_args.push_back(arg);
            continue;
        }
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--print-config")
        {
            print_config = true;
        }
        else if (arg == "--calibrate-aim")
        {
            cli.calibrate_aim = true;
        }
        else if (arg == "--calibrate-zero")
        {
            cli.calibrate_zero = true;
        }
        else if (arg == "--aim-flip")
        {
            cli.aim_flip = true;
        }
        else if (const auto value = flag_value(arg, "config"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "config");
            }
            config_path = *value;
        }
        else if (const auto value = flag_value(arg, "joints-topic"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "joints-topic");
            }
            cli.joints_topic = *value;
        }
        else if (const auto value = flag_value(arg, "links-topic"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "links-topic");
            }
            cli.links_topic = *value;
        }
        else if (const auto value = flag_value(arg, "joints-left-collection-id"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "joints-left-collection-id");
            }
            cli.joints_left_collection_id = *value;
        }
        else if (const auto value = flag_value(arg, "joints-right-collection-id"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "joints-right-collection-id");
            }
            cli.joints_right_collection_id = *value;
        }
        else if (const auto value = flag_value(arg, "tips-left-collection-id"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "tips-left-collection-id");
            }
            cli.tips_left_collection_id = *value;
        }
        else if (const auto value = flag_value(arg, "tips-right-collection-id"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "tips-right-collection-id");
            }
            cli.tips_right_collection_id = *value;
        }
        else if (const auto value = flag_value(arg, "tips-reference-frame"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "tips-reference-frame");
            }
            cli.tips_reference_frame = *value;
        }
        else if (const auto value = flag_value(arg, "wrist-source"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "wrist-source");
            }
            cli.wrist_source = *value;
        }
        else if (const auto value = flag_value(arg, "left-aim-to-wrist"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "left-aim-to-wrist");
            }
            cli.left_aim_to_wrist = *value;
        }
        else if (const auto value = flag_value(arg, "right-aim-to-wrist"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "right-aim-to-wrist");
            }
            cli.right_aim_to_wrist = *value;
        }
        else if (const auto value = flag_value(arg, "wrist-smoothing-ms"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "wrist-smoothing-ms");
            }
            try
            {
                cli.wrist_smoothing_ms = std::stod(*value);
            }
            catch (const std::exception&)
            {
                std::cerr << "Error: --wrist-smoothing-ms= must be a number." << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        }
        else if (const auto value = flag_value(arg, "aim-behind"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "aim-behind");
            }
            double metres = 0.0;
            if (!parse_metres(*value, "aim-behind", argv[0], metres))
            {
                return 1;
            }
            cli.aim_behind_m = metres;
        }
        else if (const auto value = flag_value(arg, "aim-height"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "aim-height");
            }
            double metres = 0.0;
            if (!parse_metres(*value, "aim-height", argv[0], metres))
            {
                return 1;
            }
            cli.aim_height_m = metres;
        }
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Config file first, then the environment, then the command line on top, so an ad-hoc flag beats
    // everything else. The capture's --wrist-source is a label the plugin cannot see, so it is read
    // from the environment variable the capture sets alongside it.
    const ConfigResolution resolution = load_options_from_config(config_path, options);
    if (const char* from_env = std::getenv(kWristSourceEnvVar); from_env != nullptr && *from_env != '\0')
    {
        options.wrist_source = from_env;
    }
    if (cli.joints_topic)
    {
        options.joints_topic = *cli.joints_topic;
    }
    if (cli.links_topic)
    {
        options.links_topic = *cli.links_topic;
    }
    if (cli.joints_left_collection_id)
    {
        options.joints_left_collection_id = *cli.joints_left_collection_id;
    }
    if (cli.joints_right_collection_id)
    {
        options.joints_right_collection_id = *cli.joints_right_collection_id;
    }
    if (cli.tips_left_collection_id)
    {
        options.tips_left_collection_id = *cli.tips_left_collection_id;
    }
    if (cli.tips_right_collection_id)
    {
        options.tips_right_collection_id = *cli.tips_right_collection_id;
    }
    if (cli.tips_reference_frame)
    {
        options.tips_reference_frame = *cli.tips_reference_frame;
    }
    if (cli.wrist_source)
    {
        options.wrist_source = *cli.wrist_source;
    }
    if (cli.left_aim_to_wrist)
    {
        options.left_aim_to_wrist = *cli.left_aim_to_wrist;
    }
    if (cli.right_aim_to_wrist)
    {
        options.right_aim_to_wrist = *cli.right_aim_to_wrist;
    }
    if (cli.wrist_smoothing_ms)
    {
        options.wrist_smoothing_ms = *cli.wrist_smoothing_ms;
    }

    if (print_config)
    {
        std::cout << "Config file: "
                  << (resolution.loaded_path.empty() ? "<none found, using built-in defaults>" : resolution.loaded_path)
                  << "\nSearched:\n";
        for (const auto& candidate : resolution.searched_paths)
        {
            std::cout << "  " << candidate << "\n";
        }
        std::cout << "joints_topic            = " << options.joints_topic << "\n"
                  << "links_topic             = " << options.links_topic << "\n"
                  << "joints_left_collection_id  = " << options.joints_left_collection_id << "\n"
                  << "joints_right_collection_id = " << options.joints_right_collection_id << "\n"
                  << "tips_left_collection_id = " << options.tips_left_collection_id << "\n"
                  << "tips_right_collection_id= " << options.tips_right_collection_id << "\n"
                  << "tips_reference_frame    = " << options.tips_reference_frame << "\n"
                  << "wrist_source            = " << options.wrist_source << "\n"
                  << "left_aim_to_wrist       = " << options.left_aim_to_wrist << "\n"
                  << "right_aim_to_wrist      = " << options.right_aim_to_wrist << "\n"
                  << "left_joint_zero_offsets  = " << zero_offsets_text(options.left_joint_zero_offsets) << "\n"
                  << "right_joint_zero_offsets = " << zero_offsets_text(options.right_joint_zero_offsets) << "\n"
                  << "wrist_smoothing_ms      = " << options.wrist_smoothing_ms << "\n"
                  << "joints buffer (bytes)   = " << IoExoGloveRos2Plugin::kJointsMaxFlatbufferSize << "\n"
                  << "tips buffer (bytes)     = " << kTipMaxFlatbufferSize
                  << "\n(Both sizes are the consumer trackers' own defaults -- JointStateTracker and "
                     "JointSe3PoseTracker -- so no explicit size is needed for either collection.)\n"
                     "(left/right_joint_zero_offsets: 21 whitespace- or comma-separated degrees,\n"
                     " in the plugin's joint order -- thumb F1..F4, index..little F1..F4, then ThumbBase;\n"
                     " empty = baked table)"
                  << std::endl;
        return 0;
    }

    if (!resolution.loaded_path.empty())
    {
        logger()->info("Loaded config: {}", resolution.loaded_path);
    }

    // The two one-shot modes measure different things and print different config lines; running both
    // would make one of the two windows meaningless, so reject the combination.
    if (cli.calibrate_aim && cli.calibrate_zero)
    {
        logger()->error("--calibrate-aim and --calibrate-zero are mutually exclusive");
        return 1;
    }
    // This mode writes the aim-to-wrist constants for the controller source; any other source has no
    // aim pose to calibrate, so fail before creating a session rather than after.
    if (cli.calibrate_aim && options.wrist_source != "controller")
    {
        logger()->error("--calibrate-aim requires --wrist-source=controller, not '{}'", options.wrist_source);
        return 1;
    }
    // The flip only affects the dorsal hint the one-shot calibration builds; outside that mode it
    // would silently do nothing, so fail loudly instead.
    if (cli.aim_flip && !cli.calibrate_aim)
    {
        logger()->error("--aim-flip is only meaningful with --calibrate-aim");
        return 1;
    }

    // The calibration measures the raw controller aim pose, so the wrist source must not have the
    // configured offset folded in (WristPoseSource::query() applies it). Identity here is for this
    // one-shot run only; the configured values are what the printed ones replace.
    IoExoGloveRos2Plugin::Options run_options = options;
    if (cli.calibrate_aim)
    {
        run_options.left_aim_to_wrist = "0,0,0,0,0,0,1";
        run_options.right_aim_to_wrist = "0,0,0,0,0,0,1";
        logger()->info("Aim calibration: wrist {:.3f} m behind and {:.3f} m below the controller aim origin{}",
                       cli.aim_behind_m.value_or(kDefaultAimBehindM), cli.aim_height_m.value_or(kDefaultAimHeightM),
                       cli.aim_flip ? ", dorsal hint flipped" : "");
    }

    // ROS gets only the arguments it owns: our own flags and the launcher-injected
    // --plugin-root-id have already been consumed.
    std::vector<char*> ros_argv;
    ros_argv.reserve(ros_args.size() + 1);
    for (std::string& arg : ros_args)
    {
        ros_argv.push_back(arg.data());
    }
    ros_argv.push_back(nullptr);
    rclcpp::init(static_cast<int>(ros_args.size()), ros_argv.data());

    logger()->info("IoExoGloveRos2Plugin (joints: {} -> {}, {}, tips: {} -> {} / {})", options.joints_topic,
                   options.joints_left_collection_id, options.joints_right_collection_id, options.links_topic,
                   options.tips_left_collection_id, options.tips_right_collection_id);

    auto node = std::make_shared<IoExoGloveRos2Plugin>(run_options);
    if (cli.calibrate_aim)
    {
        node->calibrate_aim(static_cast<float>(cli.aim_behind_m.value_or(kDefaultAimBehindM)),
                            static_cast<float>(cli.aim_height_m.value_or(kDefaultAimHeightM)), cli.aim_flip);
        rclcpp::shutdown();
        return 0;
    }
    if (cli.calibrate_zero)
    {
        logger()->info("Zero calibration: waiting for the glove link stream on '{}'", options.links_topic);
        node->calibrate_zero();
        rclcpp::shutdown();
        return 0;
    }
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
catch (const std::exception& e)
{
    logger()->error("{}: {}", argv[0], e.what());
    return 1;
}
catch (...)
{
    logger()->error("{}: Unknown error", argv[0]);
    return 1;
}
