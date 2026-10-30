/**
 * The fixed voice-command grammar: a pure function from one final recognition
 * result string to a recognized intent, or `null` for anything that doesn't
 * match. Matches the ENTIRE trimmed/normalized utterance -- never a command
 * word found inside an arbitrary sentence, and never an interim (non-final)
 * result, both of which are the caller's responsibility to have already
 * filtered out before calling this.
 */

export type VoiceIntent =
  | { kind: "record" }
  | { kind: "stop" }
  // Discard itself is never dispatched straight from voice -- it opens a
  // local (click-based) confirm/cancel prompt; see RecordingPanel.tsx.
  | { kind: "discard" }
  | { kind: "task"; text: string }
  | { kind: "saveTask" }
  | { kind: "cancelTask" }
  | { kind: "stopListening" };

/** Lowercases, trims, collapses internal whitespace, and drops one trailing
 * harmless punctuation mark (".", "!", "?") -- browser recognizers commonly
 * append one. Does not touch punctuation inside a `task <text>` argument,
 * which keeps its own case/punctuation as recognized (see parseVoicePhrase). */
function normalize(raw: string): string {
  return raw.trim().replace(/\s+/g, " ").replace(/[.!?]+$/u, "").toLowerCase();
}

const RECORD_PHRASES = new Set(["record", "start recording", "begin recording"]);
// "done" deliberately excluded: with an always-on mic it's one of the most common words in
// ordinary conversation, far likelier to false-trigger a Stop than the other phrases here.
const STOP_PHRASES = new Set(["stop", "stop recording", "save recording"]);
const DISCARD_PHRASES = new Set(["discard", "discard recording"]);
const SAVE_TASK_PHRASES = new Set(["save task"]);
const CANCEL_TASK_PHRASES = new Set(["cancel task"]);
const STOP_LISTENING_PHRASES = new Set(["stop listening"]);

const TASK_PREFIXES = ["task ", "set task "];

/** Parses one already-final recognition result into a {@link VoiceIntent}, or
 * `null` if it matches nothing in the fixed grammar. */
export function parseVoicePhrase(raw: string): VoiceIntent | null {
  const normalized = normalize(raw);
  if (normalized === "") return null;

  if (RECORD_PHRASES.has(normalized)) return { kind: "record" };
  if (STOP_PHRASES.has(normalized)) return { kind: "stop" };
  if (DISCARD_PHRASES.has(normalized)) return { kind: "discard" };
  if (SAVE_TASK_PHRASES.has(normalized)) return { kind: "saveTask" };
  if (CANCEL_TASK_PHRASES.has(normalized)) return { kind: "cancelTask" };
  if (STOP_LISTENING_PHRASES.has(normalized)) return { kind: "stopListening" };

  // The task argument is taken from the ORIGINAL (not lowercased) trimmed
  // text, matched by prefix length on the normalized form, so the spoken
  // task text keeps whatever case/punctuation the recognizer produced --
  // only the command prefix itself is normalized for matching.
  const trimmedRaw = raw.trim().replace(/\s+/g, " ");
  for (const prefix of TASK_PREFIXES) {
    if (normalized.startsWith(prefix)) {
      const text = trimmedRaw.slice(prefix.length).trim();
      return text === "" ? null : { kind: "task", text };
    }
  }
  return null;
}
