// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_config.hpp"
#include "io_exo_glove_ros2_pack.hpp"
#include "io_exo_glove_ros2_plugin.hpp"

#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace plugins::io_exo_glove_ros2;

namespace
{

//! Values taken from the command line. A field left unset keeps whatever the config file or the
//! built-in default provided.
struct CliOverrides
{
    std::optional<std::string> joints_topic;
    std::optional<std::string> links_topic;
    std::optional<std::string> joints_collection_id;
    std::optional<std::string> tips_left_collection_id;
    std::optional<std::string> tips_right_collection_id;
    std::optional<std::string> tips_reference_frame;
};

void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options] [--ros-args ...]\n"
              << "\nTopics, collection ids and the fingertip reference frame come from the config file; the flags\n"
              << "below override it.\n"
              << "\nOptions:\n"
              << "  --config=PATH                  Config file to load (default: the default path, or $"
              << kConfigEnvVar << ")\n"
              << "  --joints-topic=TOPIC           Override the glove joint angles topic (sensor_msgs/JointState)\n"
              << "  --links-topic=TOPIC            Override the exoskeleton link poses topic (tf2_msgs/TFMessage)\n"
              << "  --joints-collection-id=ID      Override the joint angles tensor collection id\n"
              << "  --tips-left-collection-id=ID   Override the left fingertip poses collection id\n"
              << "  --tips-right-collection-id=ID  Override the right fingertip poses collection id\n"
              << "  --tips-reference-frame=FRAME   Override the frame the fingertip poses are expressed in\n"
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
        else if (const auto value = flag_value(arg, "joints-collection-id"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "joints-collection-id");
            }
            cli.joints_collection_id = *value;
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
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Config file first, then the command line on top, so an ad-hoc flag beats the file.
    const ConfigResolution resolution = load_options_from_config(config_path, options);
    if (cli.joints_topic)
    {
        options.joints_topic = *cli.joints_topic;
    }
    if (cli.links_topic)
    {
        options.links_topic = *cli.links_topic;
    }
    if (cli.joints_collection_id)
    {
        options.joints_collection_id = *cli.joints_collection_id;
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
                  << "joints_collection_id    = " << options.joints_collection_id << "\n"
                  << "tips_left_collection_id = " << options.tips_left_collection_id << "\n"
                  << "tips_right_collection_id= " << options.tips_right_collection_id << "\n"
                  << "tips_reference_frame    = " << options.tips_reference_frame << "\n"
                  << "joints buffer (bytes)   = " << IoExoGloveRos2Plugin::kJointsMaxFlatbufferSize << "\n"
                  << "tips buffer (bytes)     = " << kTipMaxFlatbufferSize
                  << "\n(Both sizes are the consumer trackers' own defaults -- JointStateTracker and "
                     "JointSe3PoseTracker -- so no explicit size is needed for either collection.)"
                  << std::endl;
        return 0;
    }

    if (!resolution.loaded_path.empty())
    {
        std::cout << "Loaded config: " << resolution.loaded_path << std::endl;
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

    std::cout << "IoExoGloveRos2Plugin (joints: " << options.joints_topic << " -> " << options.joints_collection_id
              << ", tips: " << options.links_topic << " -> " << options.tips_left_collection_id << " / "
              << options.tips_right_collection_id << ")" << std::endl;

    auto node = std::make_shared<IoExoGloveRos2Plugin>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << argv[0] << ": Unknown error" << std::endl;
    return 1;
}
