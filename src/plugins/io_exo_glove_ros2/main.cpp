// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_config.hpp"
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
    std::optional<std::string> left_topic;
    std::optional<std::string> right_topic;
    std::optional<std::string> left_collection_id;
    std::optional<std::string> right_collection_id;
};

void print_usage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options] [--ros-args ...]\n"
              << "\nTopics and collection ids come from the config file; the flags below override it.\n"
              << "\nOptions:\n"
              << "  --config=PATH               Config file to load (default: the default path, or $" << kConfigEnvVar
              << ")\n"
              << "  --left-topic=TOPIC          Override the left hand's ROS 2 JointState topic\n"
              << "  --right-topic=TOPIC         Override the right hand's ROS 2 JointState topic\n"
              << "  --left-collection-id=ID     Override the left OpenXR tensor collection id\n"
              << "  --right-collection-id=ID    Override the right OpenXR tensor collection id\n"
              << "  --print-config              Print the resolved settings and the file they came from, then exit\n"
              << "  --help                      Show this help\n"
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

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--ros-args")
        {
            // ROS 2 owns the remainder (remappings, parameter files, logging): forward it verbatim
            // and stop looking for our own flags.
            ros_args.insert(ros_args.end(), argv + i, argv + argc);
            break;
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
        else if (const auto value = flag_value(arg, "left-topic"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "left-topic");
            }
            cli.left_topic = *value;
        }
        else if (const auto value = flag_value(arg, "right-topic"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "right-topic");
            }
            cli.right_topic = *value;
        }
        else if (const auto value = flag_value(arg, "left-collection-id"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "left-collection-id");
            }
            cli.left_collection_id = *value;
        }
        else if (const auto value = flag_value(arg, "right-collection-id"))
        {
            if (value->empty())
            {
                return fail_empty_flag(argv[0], "right-collection-id");
            }
            cli.right_collection_id = *value;
        }
        else if (arg.rfind("--plugin-root-id=", 0) == 0)
        {
            // Injected by the PluginManager for every plugin it launches; unused by this transport
            // bridge (which publishes under fixed collection ids) but must not be an error.
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
    if (cli.left_topic)
    {
        options.left_topic = *cli.left_topic;
    }
    if (cli.right_topic)
    {
        options.right_topic = *cli.right_topic;
    }
    if (cli.left_collection_id)
    {
        options.left_collection_id = *cli.left_collection_id;
    }
    if (cli.right_collection_id)
    {
        options.right_collection_id = *cli.right_collection_id;
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
        std::cout << "left_topic          = " << options.left_topic << "\n"
                  << "right_topic         = " << options.right_topic << "\n"
                  << "left_collection_id  = " << options.left_collection_id << "\n"
                  << "right_collection_id = " << options.right_collection_id << std::endl;
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

    std::cout << "IoExoGloveRos2Plugin (left: " << options.left_topic << " -> " << options.left_collection_id
              << ", right: " << options.right_topic << " -> " << options.right_collection_id << ")" << std::endl;

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
