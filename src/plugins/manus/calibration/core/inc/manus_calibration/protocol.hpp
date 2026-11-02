// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// Newline-delimited JSON wire format between the calibration worker and its supervisor.
// The field contract is documented in src/plugins/manus/calibration/README.md.
namespace plugins::manus::calibration
{

inline constexpr int kProtocolVersion = 1;
inline constexpr size_t kMaxMessageBytes = 65536;

enum class Side
{
    Left,
    Right,
};

std::string_view side_name(Side side);

struct StartCommand
{
    std::string attempt;
    // Always in processing order: left before right.
    std::vector<Side> sides;
    std::string staging_dir;
};

struct ConfirmCommand
{
    std::string attempt;
    std::string token;
};

struct CancelCommand
{
    std::string attempt;
};

using Command = std::variant<StartCommand, ConfirmCommand, CancelCommand>;

enum class RejectReason
{
    Invalid,
    UnknownAttempt,
    StaleToken,
    Unexpected,
    Busy,
};

enum class Phase
{
    Discovering,
    AwaitingStep,
    CapturingStep,
    Saving,
    Stopping,
};

enum class StepMode
{
    Hold,
    Motion,
};

enum class Outcome
{
    Completed,
    Cancelled,
    Failed,
};

enum class ErrorCode
{
    SdkInitFailed,
    NoGloves,
    GloveMissing,
    GloveReplaced,
    SdkError,
    StepFailed,
    ExportFailed,
    ProtocolError,
    Internal,
};

struct GloveRef
{
    Side side;
    uint32_t glove_id;
};

struct ReadyEvent
{
    std::string worker_version;
    std::optional<std::string> sdk_version;
};

struct StatusEvent
{
    std::string attempt;
    Phase phase;
    std::vector<GloveRef> gloves;
};

struct PromptEvent
{
    std::string attempt;
    std::string token;
    Side side;
    uint32_t glove_id;
    uint32_t index;
    uint32_t count;
    std::string title;
    std::string instruction;
    StepMode mode;
    double duration_s;
    // True for an SDK step with a negative time. It is operator-paced like a hold; estimated_s is the
    // SDK's estimate of how long the operator keeps the pose or movement, and zero otherwise.
    bool continuous = false;
    double estimated_s = 0.0;
};

struct StepAcceptedEvent
{
    std::string attempt;
    std::string token;
};

struct StepProgressEvent
{
    std::string attempt;
    std::string token;
    double elapsed_s;
};

struct RejectedEvent
{
    std::optional<std::string> attempt;
    std::string command;
    RejectReason reason;
    std::string message;
    std::optional<std::string> token;
};

struct ExportedEvent
{
    std::string attempt;
    Side side;
    uint32_t glove_id;
    std::string file;
    uint64_t bytes;
    std::string sha256;
};

struct ResultFile
{
    Side side;
    std::string file;
    uint64_t bytes;
    std::string sha256;
};

struct ErrorInfo
{
    ErrorCode code;
    std::string message;
    std::optional<Side> side;
    std::optional<uint32_t> step;
};

struct ResultEvent
{
    std::optional<std::string> attempt;
    Outcome outcome;
    std::vector<ResultFile> files;
    std::optional<ErrorInfo> error;
};

using Event = std::variant<ReadyEvent,
                           StatusEvent,
                           PromptEvent,
                           StepAcceptedEvent,
                           StepProgressEvent,
                           RejectedEvent,
                           ExportedEvent,
                           ResultEvent>;

// Exactly one of the two members is set.
struct ParsedCommand
{
    std::optional<Command> command;
    std::optional<RejectedEvent> rejection;
};

// Strict: unknown keys, wrong types, bad values and oversized lines all yield a rejection.
ParsedCommand parse_command(std::string_view line);

// One JSON object without the trailing newline.
std::string serialize_event(const Event& event);

} // namespace plugins::manus::calibration
