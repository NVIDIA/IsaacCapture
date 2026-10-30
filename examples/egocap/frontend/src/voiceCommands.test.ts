import { describe, expect, it } from "vitest";
import { parseVoicePhrase } from "./voiceCommands";

describe("parseVoicePhrase", () => {
  it.each([
    ["record", { kind: "record" }],
    ["Record.", { kind: "record" }],
    ["start recording", { kind: "record" }],
    ["begin recording", { kind: "record" }],
    ["stop", { kind: "stop" }],
    ["stop recording", { kind: "stop" }],
    ["save recording", { kind: "stop" }],
    ["discard", { kind: "discard" }],
    ["discard recording", { kind: "discard" }],
    ["save task", { kind: "saveTask" }],
    ["cancel task", { kind: "cancelTask" }],
    ["stop listening", { kind: "stopListening" }],
  ])("recognizes %j as %j", (phrase, expected) => {
    expect(parseVoicePhrase(phrase)).toEqual(expected);
  });

  it("extracts the task argument, preserving its original case/punctuation", () => {
    expect(parseVoicePhrase("task Dry the Dishes!")).toEqual({ kind: "task", text: "Dry the Dishes!" });
    expect(parseVoicePhrase("set task fold the towel")).toEqual({ kind: "task", text: "fold the towel" });
  });

  it("normalizes whitespace and a trailing punctuation mark before matching the command prefix", () => {
    expect(parseVoicePhrase("  Stop   Recording.  ")).toEqual({ kind: "stop" });
  });

  it("does not match a command word found inside an arbitrary sentence", () => {
    expect(parseVoicePhrase("please stop bothering me")).toBeNull();
    expect(parseVoicePhrase("I will record this later")).toBeNull();
  });

  it('does not recognize "done" -- one of the most common words in ordinary conversation with an always-on mic', () => {
    expect(parseVoicePhrase("done")).toBeNull();
  });

  it("rejects an empty or bare task command with no argument", () => {
    expect(parseVoicePhrase("")).toBeNull();
    expect(parseVoicePhrase("   ")).toBeNull();
    expect(parseVoicePhrase("task")).toBeNull();
    expect(parseVoicePhrase("task   ")).toBeNull();
  });

  it("rejects unrelated sentences entirely", () => {
    expect(parseVoicePhrase("what time is it")).toBeNull();
    expect(parseVoicePhrase("discard everything now please")).toBeNull();
  });
});
