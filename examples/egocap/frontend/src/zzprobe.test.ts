import { it, expect } from "vitest";
import { useLiveKitPreview } from "./useLiveKitPreview";
it("loads", () => expect(typeof useLiveKitPreview).toBe("function"));
