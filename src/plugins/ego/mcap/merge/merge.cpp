// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/ego_mcap/merge.hpp"

#include <mcap/reader.hpp>
#include <mcap/writer.hpp>

#include <array>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace plugins::ego
{
namespace
{
constexpr std::string_view observed_prefix = "session_observed/";
constexpr std::array structured_names = { "core.EgoFrameMetadataRecord", "core.EgoImuBatchRecord",
                                          "core.EgoAudioChunkRecord", "core.EgoCalibrationRecord",
                                          "core.EgoDeviceStateRecord" };

void check(const mcap::Status& status)
{
    if (!status.ok())
        throw std::runtime_error("MCAP merge: " + status.message);
}

struct Counts
{
    uint64_t messages = 0;
    uint64_t metadata = 0;
    uint64_t attachments = 0;
};

struct Source
{
    mcap::McapReader reader;
    std::unordered_map<mcap::SchemaId, mcap::SchemaPtr> schemas;
    std::unordered_map<mcap::ChannelId, mcap::ChannelPtr> channels;
    std::unordered_map<mcap::ChannelId, mcap::ChannelId> output_channels;
    Counts counts;
    std::optional<bool> record_device_diagnostics;

    explicit Source(const std::filesystem::path& path, bool reject_observed = true)
    {
        check(reader.open(path.string()));
        check(reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan));
        if (!reader.header() || !reader.footer() || !reader.statistics())
            throw std::runtime_error("MCAP merge requires a complete file with a summary: " + path.string());
        schemas = reader.schemas();
        channels = reader.channels();
        for (const auto& [id, channel] : channels)
        {
            if (id == 0 || (channel->schemaId != 0 && !schemas.contains(channel->schemaId)))
                throw std::runtime_error("MCAP channel references a missing schema");
            if (reject_observed && channel->topic.starts_with(observed_prefix))
                throw std::runtime_error("Reserved MCAP topic prefix already exists: " + channel->topic);
        }
    }

    mcap::SchemaPtr schema(const mcap::Channel& channel) const
    {
        return channel.schemaId == 0 ? nullptr : schemas.at(channel.schemaId);
    }
};

bool same_schema(const Source& left, const mcap::Channel& a, const Source& right, const mcap::Channel& b)
{
    if (a.messageEncoding != b.messageEncoding)
        return false;
    const auto sa = left.schema(a);
    const auto sb = right.schema(b);
    if (!sa || !sb)
        return !sa && !sb;
    return sa->name == sb->name && sa->encoding == sb->encoding && sa->data == sb->data;
}

bool structured(const Source& source, const mcap::Channel& channel)
{
    const auto schema = source.schema(channel);
    if (!schema)
        return false;
    for (const auto name : structured_names)
        if (schema->name == name)
            return true;
    return false;
}

Counts scan(Source& source, mcap::McapWriter* output = nullptr)
{
    Counts counts;
    unsigned headers = 0;
    unsigned footers = 0;
    unsigned ends = 0;
    auto& data = *source.reader.dataSource();
    mcap::TypedRecordReader records(data, sizeof(mcap::Magic), data.size() - sizeof(mcap::Magic));
    records.onHeader = [&](const auto&, auto) { ++headers; };
    records.onFooter = [&](const auto&, auto) { ++footers; };
    records.onDataEnd = [&](const auto&, auto) { ++ends; };
    records.onSchema = [&](const mcap::SchemaPtr schema, auto, auto)
    {
        const auto found = source.schemas.find(schema->id);
        if (found == source.schemas.end() || found->second->name != schema->name ||
            found->second->encoding != schema->encoding || found->second->data != schema->data)
            throw std::runtime_error("MCAP data and summary schemas disagree");
    };
    records.onChannel = [&](const mcap::ChannelPtr channel, auto, auto)
    {
        const auto found = source.channels.find(channel->id);
        if (found == source.channels.end() || found->second->topic != channel->topic ||
            found->second->schemaId != channel->schemaId ||
            found->second->messageEncoding != channel->messageEncoding || found->second->metadata != channel->metadata)
            throw std::runtime_error("MCAP data and summary channels disagree");
    };
    records.onMessage = [&](const mcap::Message& message, auto, auto)
    {
        if (!source.channels.contains(message.channelId))
            throw std::runtime_error("MCAP message references an unknown channel");
        if (message.logTime == mcap::MaxTime)
            throw std::runtime_error("MCAP logTime exceeds the indexed reader's range");
        ++counts.messages;
    };
    records.onMetadata = [&](const mcap::Metadata& metadata, auto)
    {
        ++counts.metadata;
        if (metadata.name == "ego.recording")
        {
            const auto flag = metadata.metadata.find("record_device_diagnostics");
            const auto version = metadata.metadata.find("diagnostic_version");
            if (flag == metadata.metadata.end() || (flag->second != "on" && flag->second != "off") ||
                version == metadata.metadata.end() || version->second != "2")
                throw std::runtime_error("MCAP contains an unsupported EGO recording policy");
            const bool enabled = flag->second == "on";
            if (source.record_device_diagnostics && *source.record_device_diagnostics != enabled)
                throw std::runtime_error("MCAP contains conflicting EGO recording policies");
            source.record_device_diagnostics = enabled;
        }
        if (output)
            check(output->write(metadata));
    };
    records.onAttachment = [&](const mcap::Attachment& attachment, auto)
    {
        ++counts.attachments;
        if (output)
        {
            auto copy = attachment;
            check(output->write(copy));
        }
    };
    // Source indexes contain file offsets; rebuild them for the merged layout.
    records.onMessageIndex = [](const auto&, auto) {};
    records.onChunkIndex = [](const auto&, auto) {};
    records.onMetadataIndex = [](const auto&, auto) {};
    records.onAttachmentIndex = [](const auto&, auto) {};
    records.onStatistics = [](const auto&, auto) {};
    records.onSummaryOffset = [](const auto&, auto) {};
    records.onUnknownRecord = [](const auto&, auto, auto)
    { throw std::runtime_error("MCAP merge cannot preserve an unknown record type"); };
    while (records.next())
        check(records.status());
    check(records.status());
    const auto& stats = *source.reader.statistics();
    if (headers != 1 || footers != 1 || ends != 1 || counts.messages != stats.messageCount ||
        counts.metadata != stats.metadataCount || counts.attachments != stats.attachmentCount)
        throw std::runtime_error("MCAP record inventory does not match the complete file summary");
    return counts;
}

std::unordered_set<std::string> conflicting_topics(const Source& recording, const Source& media)
{
    std::unordered_set<std::string> conflicts;
    for (const Source* source : { &recording, &media })
    {
        std::unordered_map<std::string, mcap::ChannelPtr> topics;
        for (const auto& [id, channel] : source->channels)
        {
            const auto [it, inserted] = topics.emplace(channel->topic, channel);
            if (!inserted && !same_schema(*source, *it->second, *source, *channel))
                throw std::runtime_error("Conflicting schemas for MCAP topic: " + channel->topic);
        }
    }
    for (const auto& [id, host] : recording.channels)
        for (const auto& [other_id, capture] : media.channels)
            if (host->topic == capture->topic)
            {
                if (!same_schema(recording, *host, media, *capture) || !structured(media, *capture))
                    throw std::runtime_error("Conflicting MCAP topic: " + host->topic);
                conflicts.insert(host->topic);
            }
    return conflicts;
}

void register_channels(Source& source, mcap::McapWriter& output, const std::unordered_set<std::string>& observed)
{
    std::unordered_map<mcap::SchemaId, mcap::SchemaId> ids;
    for (const auto& [id, original] : source.schemas)
    {
        mcap::Schema schema(original->name, original->encoding, original->data);
        output.addSchema(schema);
        ids.emplace(id, schema.id);
    }
    for (const auto& [id, original] : source.channels)
    {
        auto topic = original->topic;
        if (observed.contains(topic))
            topic = std::string(observed_prefix) + topic;
        mcap::Channel channel(topic, original->messageEncoding,
                              original->schemaId == 0 ? 0 : ids.at(original->schemaId), original->metadata);
        output.addChannel(channel);
        source.output_channels.emplace(id, channel.id);
    }
}

uint64_t merge_messages(Source& recording, Source& media, mcap::McapWriter& output)
{
    mcap::ReadMessageOptions options;
    options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
    const auto problem = [](const mcap::Status& status) { check(status); };
    auto host_messages = recording.reader.readMessages(problem, options);
    auto capture_messages = media.reader.readMessages(problem, options);
    auto host = host_messages.begin();
    auto capture = capture_messages.begin();
    uint64_t count = 0;
    while (host != host_messages.end() || capture != capture_messages.end())
    {
        const bool take_host = capture == capture_messages.end() ||
                               (host != host_messages.end() && host->message.logTime <= capture->message.logTime);
        auto& iterator = take_host ? host : capture;
        auto& source = take_host ? recording : media;
        auto message = iterator->message;
        message.channelId = source.output_channels.at(message.channelId);
        check(output.write(message));
        ++count;
        ++iterator;
    }
    return count;
}
} // namespace

void merge_recording(const std::filesystem::path& recording_path,
                     const std::filesystem::path& media_path,
                     const std::filesystem::path& output_path)
{
    const auto recording_file = std::filesystem::weakly_canonical(recording_path);
    const auto media_file = std::filesystem::weakly_canonical(media_path);
    const auto final_file = std::filesystem::weakly_canonical(output_path);
    const auto partial_file = std::filesystem::path(final_file.string() + ".partial");
    if (recording_file == media_file || recording_file == final_file || media_file == final_file)
        throw std::invalid_argument("MCAP merge inputs and output must be distinct");
    for (const auto& path : { final_file, partial_file })
        if (std::filesystem::exists(path) || std::filesystem::is_symlink(path))
            throw std::runtime_error("Refusing to overwrite MCAP output: " + path.string());
    if (!std::filesystem::is_directory(final_file.parent_path()))
        throw std::runtime_error("Create the MCAP output directory before merging");

    Source recording(recording_file);
    Source media(media_file);
    recording.counts = scan(recording);
    media.counts = scan(media);
    if (recording.record_device_diagnostics && media.record_device_diagnostics &&
        recording.record_device_diagnostics != media.record_device_diagnostics)
        throw std::runtime_error("Parent and fragment EGO recording policies disagree");
    const auto policy =
        media.record_device_diagnostics ? media.record_device_diagnostics : recording.record_device_diagnostics;
    if (policy && !*policy)
    {
        // Match schema identity, including custom topic names and empty registered channels.
        for (const Source* source : { &recording, &media })
            for (const auto& [id, channel] : source->channels)
                if (const auto schema = source->schema(*channel); schema && schema->name == "core.EgoDeviceStateRecord")
                    throw std::runtime_error("Diagnostic recording is off but an input contains EgoDeviceState: " +
                                             channel->topic);
    }
    const auto conflicts = conflicting_topics(recording, media);
    if (recording.schemas.size() + media.schemas.size() > std::numeric_limits<mcap::SchemaId>::max() ||
        recording.channels.size() + media.channels.size() > std::numeric_limits<mcap::ChannelId>::max())
        throw std::runtime_error("Merged MCAP exceeds schema or channel ID capacity");

    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(partial_file, std::ios::binary | std::ios::out);
    mcap::McapWriter output;
    try
    {
        mcap::McapWriterOptions options(recording.reader.header()->profile);
        options.compression = mcap::Compression::None;
        options.library = "isaaccapture.ego.merge";
        output.open(stream, options);
        // The fragment includes capture-tail samples that the host may not poll.
        // Preserve overlapping host observations under session_observed/.
        register_channels(recording, output, conflicts);
        register_channels(media, output, {});
        scan(recording, &output);
        scan(media, &output);
        mcap::Metadata provenance;
        provenance.name = "ego.merge";
        provenance.metadata = { { "recording", recording_file.string() },
                                { "media", media_file.string() },
                                { "recording.library", recording.reader.header()->library },
                                { "media.library", media.reader.header()->library },
                                { "media.profile", media.reader.header()->profile },
                                { "capture.channels", "complete plugin capture" },
                                { "session_observed.channels", "host session observations" } };
        check(output.write(provenance));
        const auto expected_messages = recording.counts.messages + media.counts.messages;
        if (merge_messages(recording, media, output) != expected_messages)
            throw std::runtime_error("Indexed MCAP merge did not preserve every message");
        output.close();
        stream.close();
        Source validation(partial_file, false);
        const auto counts = scan(validation);
        if (counts.messages != expected_messages ||
            counts.metadata != recording.counts.metadata + media.counts.metadata + 1 ||
            counts.attachments != recording.counts.attachments + media.counts.attachments)
            throw std::runtime_error("Merged MCAP failed the final inventory check");
        validation.reader.close();
        if (std::filesystem::exists(final_file) || std::filesystem::is_symlink(final_file))
            throw std::runtime_error("MCAP output appeared during merge: " + final_file.string());
        std::filesystem::rename(partial_file, final_file);
    }
    catch (...)
    {
        output.terminate();
        throw;
    }
}
} // namespace plugins::ego
