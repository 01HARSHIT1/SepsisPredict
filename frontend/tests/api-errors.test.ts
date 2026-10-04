import { describe, expect, it } from "vitest";
import { detailMessage } from "@/lib/api";

describe("API error text", () => {
  it("keeps a short validation message and drops tracebacks", () => {
    expect(detailMessage([{ msg: "Field required" }])).toBe("Field required");
    expect(detailMessage("observations must be in chronological order")).toBe(
      "observations must be in chronological order",
    );
    expect(detailMessage("Traceback (most recent call last)")).toBeNull();
  });
});
