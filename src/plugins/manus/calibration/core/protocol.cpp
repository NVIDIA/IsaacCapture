// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/manus_calibration/protocol.hpp"

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <set>

namespace plugins::manus::calibration
{

std::string_view side_name(Side side)
{
    return side == Side::Left ? "left" : "right";
}

namespace
{

constexpr size_t kMaxAttemptChars = 64;
constexpr size_t kMaxTokenChars = 128;
constexpr size_t kMaxPathChars = 4096;

bool valid_attempt(std::string_view attempt)
{
    return !attempt.empty() && attempt.size() <= kMaxAttemptChars &&
           std::all_of(attempt.begin(), attempt.end(),
                       [](unsigned char c)
                       { return std::isalnum(c) != 0 || c == '_' || c == '-'; });
}

bool valid_token(std::string_view token)
{
    return !token.empty() && token.size() <= kMaxTokenChars &&
           std::all_of(token.begin(), token.end(), [](unsigned char c) { return c > 0x20 && c < 0x7F; });
}

class FieldReader
{
public:
    explicit FieldReader(const json::Members& members) : m_members(members)
    {
    }

    const json::Value* find(std::string_view key) const
    {
        for (const auto& member : m_members)
        {
            if (member.first == key)
            {
                return &member.second;
            }
        }
        return nullptr;
    }

    // Every key must be one of `allowed`.
    std::optional<std::string> unknown_key(const std::set<std::string_view>& allowed) const
    {
        for (const auto& member : m_members)
        {
            if (allowed.count(member.first) == 0)
            {
                return member.first;
            }
        }
        return std::nullopt;
    }

    const std::string* string_field(std::string_view key) const
    {
        const json::Value* value = find(key);
        return value ? std::get_if<std::string>(value) : nullptr;
    }

private:
    const json::Members& m_members;
};

RejectedEvent invalid(std::string command, std::string message, std::optional<std::string> attempt = std::nullopt)
{
    return RejectedEvent{ std::move(attempt), std::move(command), RejectReason::Invalid, std::move(message),
                          std::nullopt };
}

ParsedCommand reject(RejectedEvent event)
{
    return ParsedCommand{ std::nullopt, std::move(event) };
}

ParsedCommand accept(Command command)
{
    return ParsedCommand{ std::move(command), std::nullopt };
}

ParsedCommand parse_start(const FieldReader& fields)
{
    static const std::set<std::string_view> allowed = { "v", "type", "attempt", "sides", "staging_dir" };
    if (const auto unknown = fields.unknown_key(allowed))
    {
        return reject(invalid("start", "unknown key \"" + *unknown + "\""));
    }
    const std::string* attempt = fields.string_field("attempt");
    if (!attempt || !valid_attempt(*attempt))
    {
        return reject(invalid("start", "attempt must be 1-64 characters of [A-Za-z0-9_-]"));
    }
    const json::Value* sides_value = fields.find("sides");
    const auto* sides_raw = sides_value ? std::get_if<std::vector<std::string>>(sides_value) : nullptr;
    if (!sides_raw || sides_raw->empty())
    {
        return reject(invalid("start", "sides must be a non-empty array", *attempt));
    }
    bool left = false;
    bool right = false;
    for (const std::string& name : *sides_raw)
    {
        if (name != "left" && name != "right")
        {
            return reject(invalid("start", "sides entries must be \"left\" or \"right\"", *attempt));
        }
        bool& seen = (name == "left") ? left : right;
        if (seen)
        {
            return reject(invalid("start", "sides must not repeat a side", *attempt));
        }
        seen = true;
    }
    const std::string* staging_dir = fields.string_field("staging_dir");
    if (!staging_dir || staging_dir->empty() || staging_dir->size() > kMaxPathChars || (*staging_dir)[0] != '/' ||
        staging_dir->find('\0') != std::string::npos)
    {
        return reject(invalid("start", "staging_dir must be an absolute path", *attempt));
    }

    StartCommand command;
    command.attempt = *attempt;
    // Fixed processing order regardless of how the supervisor listed the sides.
    if (left)
    {
        command.sides.push_back(Side::Left);
    }
    if (right)
    {
        command.sides.push_back(Side::Right);
    }
    command.staging_dir = *staging_dir;
    return accept(std::move(command));
}

ParsedCommand parse_confirm(const FieldReader& fields)
{
    static const std::set<std::string_view> allowed = { "v", "type", "attempt", "token" };
    if (const auto unknown = fields.unknown_key(allowed))
    {
        return reject(invalid("confirm", "unknown key \"" + *unknown + "\""));
    }
    const std::string* attempt = fields.string_field("attempt");
    if (!attempt || !valid_attempt(*attempt))
    {
        return reject(invalid("confirm", "attempt must be 1-64 characters of [A-Za-z0-9_-]"));
    }
    const std::string* token = fields.string_field("token");
    if (!token || !valid_token(*token))
    {
        return reject(invalid("confirm", "token must be 1-128 printable ASCII characters", *attempt));
    }
    return accept(ConfirmCommand{ *attempt, *token });
}

ParsedCommand parse_cancel(const FieldReader& fields)
{
    static const std::set<std::string_view> allowed = { "v", "type", "attempt" };
    if (const auto unknown = fields.unknown_key(allowed))
    {
        return reject(invalid("cancel", "unknown key \"" + *unknown + "\""));
    }
    const std::string* attempt = fields.string_field("attempt");
    if (!attempt || !valid_attempt(*attempt))
    {
        return reject(invalid("cancel", "attempt must be 1-64 characters of [A-Za-z0-9_-]"));
    }
    return accept(CancelCommand{ *attempt });
}

std::string_view reason_name(RejectReason reason)
{
    switch (reason)
    {
    case RejectReason::Invalid:
        return "invalid";
    case RejectReason::UnknownAttempt:
        return "unknown_attempt";
    case RejectReason::StaleToken:
        return "stale_token";
    case RejectReason::Unexpected:
        return "unexpected";
    case RejectReason::Busy:
        return "busy";
    }
    return "invalid";
}

std::string_view phase_name(Phase phase)
{
    switch (phase)
    {
    case Phase::Discovering:
        return "discovering";
    case Phase::AwaitingStep:
        return "awaiting_step";
    case Phase::CapturingStep:
        return "capturing_step";
    case Phase::Saving:
        return "saving";
    case Phase::Stopping:
        return "stopping";
    }
    return "stopping";
}

std::string_view outcome_name(Outcome outcome)
{
    switch (outcome)
    {
    case Outcome::Completed:
        return "completed";
    case Outcome::Cancelled:
        return "cancelled";
    case Outcome::Failed:
        return "failed";
    }
    return "failed";
}

std::string_view error_code_name(ErrorCode code)
{
    switch (code)
    {
    case ErrorCode::SdkInitFailed:
        return "sdk_init_failed";
    case ErrorCode::NoGloves:
        return "no_gloves";
    case ErrorCode::GloveMissing:
        return "glove_missing";
    case ErrorCode::GloveReplaced:
        return "glove_replaced";
    case ErrorCode::SdkError:
        return "sdk_error";
    case ErrorCode::StepFailed:
        return "step_failed";
    case ErrorCode::ExportFailed:
        return "export_failed";
    case ErrorCode::ProtocolError:
        return "protocol_error";
    case ErrorCode::Internal:
        return "internal";
    }
    return "internal";
}

// Builds one JSON object member by member; keeps every serializer free of manual comma bookkeeping.
class ObjectWriter
{
public:
    explicit ObjectWriter(std::string_view type)
    {
        m_out = "{\"v\":" + std::to_string(kProtocolVersion);
        string("type", type);
    }

    ObjectWriter& string(std::string_view key, std::string_view value)
    {
        key_prefix(key);
        json::append_string(m_out, value);
        return *this;
    }

    ObjectWriter& optional_string(std::string_view key, const std::optional<std::string>& value)
    {
        if (!value)
        {
            return raw(key, "null");
        }
        return string(key, *value);
    }

    ObjectWriter& integer(std::string_view key, uint64_t value)
    {
        return raw(key, std::to_string(value));
    }

    ObjectWriter& seconds(std::string_view key, double value)
    {
        // The engine rejects non-finite step times; this only keeps a line valid JSON, which has no NaN or infinity.
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.3f", std::isfinite(value) ? value : 0.0);
        return raw(key, buffer);
    }

    ObjectWriter& raw(std::string_view key, std::string_view value)
    {
        key_prefix(key);
        m_out.append(value);
        return *this;
    }

    std::string finish()
    {
        m_out.push_back('}');
        return std::move(m_out);
    }

private:
    void key_prefix(std::string_view key)
    {
        m_out.push_back(',');
        json::append_string(m_out, key);
        m_out.push_back(':');
    }

    std::string m_out;
};

std::string gloves_json(const std::vector<GloveRef>& gloves)
{
    std::string out = "[";
    for (size_t i = 0; i < gloves.size(); ++i)
    {
        if (i != 0)
        {
            out.push_back(',');
        }
        out += "{\"side\":";
        json::append_string(out, side_name(gloves[i].side));
        out += ",\"glove_id\":" + std::to_string(gloves[i].glove_id) + "}";
    }
    out.push_back(']');
    return out;
}

std::string files_json(const std::vector<ResultFile>& files)
{
    std::string out = "[";
    for (size_t i = 0; i < files.size(); ++i)
    {
        if (i != 0)
        {
            out.push_back(',');
        }
        out += "{\"side\":";
        json::append_string(out, side_name(files[i].side));
        out += ",\"file\":";
        json::append_string(out, files[i].file);
        out += ",\"bytes\":" + std::to_string(files[i].bytes) + ",\"sha256\":";
        json::append_string(out, files[i].sha256);
        out.push_back('}');
    }
    out.push_back(']');
    return out;
}

std::string error_json(const ErrorInfo& error)
{
    std::string out = "{\"code\":";
    json::append_string(out, error_code_name(error.code));
    out += ",\"message\":";
    json::append_string(out, error.message);
    if (error.side)
    {
        out += ",\"side\":";
        json::append_string(out, side_name(*error.side));
    }
    if (error.step)
    {
        out += ",\"step\":" + std::to_string(*error.step);
    }
    out.push_back('}');
    return out;
}

} // namespace

ParsedCommand parse_command(std::string_view line)
{
    if (line.size() > kMaxMessageBytes)
    {
        return reject(invalid("unknown", "message exceeds the size limit"));
    }
    const json::ParseResult parsed = json::parse_object(line);
    if (!parsed.members)
    {
        return reject(invalid("unknown", "malformed JSON: " + parsed.error));
    }
    const FieldReader fields(*parsed.members);

    const std::string* type = fields.string_field("type");
    if (!type)
    {
        return reject(invalid("unknown", "missing string field \"type\""));
    }
    const json::Value* version = fields.find("v");
    const auto* version_number = version ? std::get_if<int64_t>(version) : nullptr;
    if (!version_number || *version_number != kProtocolVersion)
    {
        return reject(invalid(*type == "start" || *type == "confirm" || *type == "cancel" ? *type : "unknown",
                              "unsupported or missing protocol version"));
    }
    if (*type == "start")
    {
        return parse_start(fields);
    }
    if (*type == "confirm")
    {
        return parse_confirm(fields);
    }
    if (*type == "cancel")
    {
        return parse_cancel(fields);
    }
    return reject(invalid("unknown", "unknown command type"));
}

std::string serialize_event(const Event& event)
{
    struct Visitor
    {
        std::string operator()(const ReadyEvent& e) const
        {
            return ObjectWriter("ready")
                .string("worker_version", e.worker_version)
                .optional_string("sdk_version", e.sdk_version)
                .finish();
        }

        std::string operator()(const StatusEvent& e) const
        {
            return ObjectWriter("status")
                .string("attempt", e.attempt)
                .string("phase", phase_name(e.phase))
                .raw("gloves", gloves_json(e.gloves))
                .finish();
        }

        std::string operator()(const PromptEvent& e) const
        {
            return ObjectWriter("prompt")
                .string("attempt", e.attempt)
                .string("token", e.token)
                .string("side", side_name(e.side))
                .integer("glove_id", e.glove_id)
                .integer("index", e.index)
                .integer("count", e.count)
                .string("title", e.title)
                .string("instruction", e.instruction)
                .string("mode", e.mode == StepMode::Motion ? "motion" : "hold")
                .seconds("duration_s", e.duration_s)
                .raw("continuous", e.continuous ? "true" : "false")
                .seconds("estimated_s", e.estimated_s)
                .finish();
        }

        std::string operator()(const StepAcceptedEvent& e) const
        {
            return ObjectWriter("step_accepted").string("attempt", e.attempt).string("token", e.token).finish();
        }

        std::string operator()(const StepProgressEvent& e) const
        {
            return ObjectWriter("step_progress")
                .string("attempt", e.attempt)
                .string("token", e.token)
                .seconds("elapsed_s", e.elapsed_s)
                .finish();
        }

        std::string operator()(const RejectedEvent& e) const
        {
            ObjectWriter writer("rejected");
            writer.optional_string("attempt", e.attempt)
                .string("command", e.command)
                .string("reason", reason_name(e.reason))
                .string("message", e.message);
            if (e.token)
            {
                writer.string("token", *e.token);
            }
            return writer.finish();
        }

        std::string operator()(const ExportedEvent& e) const
        {
            return ObjectWriter("exported")
                .string("attempt", e.attempt)
                .string("side", side_name(e.side))
                .integer("glove_id", e.glove_id)
                .string("file", e.file)
                .integer("bytes", e.bytes)
                .string("sha256", e.sha256)
                .finish();
        }

        std::string operator()(const ResultEvent& e) const
        {
            ObjectWriter writer("result");
            writer.optional_string("attempt", e.attempt).string("outcome", outcome_name(e.outcome));
            if (e.outcome == Outcome::Completed)
            {
                writer.raw("files", files_json(e.files));
            }
            if (e.error)
            {
                writer.raw("error", error_json(*e.error));
            }
            return writer.finish();
        }
    };
    return std::visit(Visitor{}, event);
}

} // namespace plugins::manus::calibration
