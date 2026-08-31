import { describe, expect, it } from "vitest";
import { formatDecision } from "@/utils/formatDecision";
import { formatEvidenceLevel, modelDisplayName } from "@/utils/formatEvidence";
import { formatUnit } from "@/utils/formatUnit";

describe("formatDecision", () => {
  it("maps all four decisions", () => {
    expect(formatDecision("RECOMMEND").label).toBe("RECOMMEND");
    expect(formatDecision("KEEP_CURRENT").label).toBe("KEEP_CURRENT");
    expect(formatDecision("REVIEW_REQUIRED").label).toBe("REVIEW_REQUIRED");
    expect(formatDecision("REJECT").label).toBe("REJECT");
  });
});

describe("formatEvidence", () => {
  it("formats model display names", () => {
    expect(modelDisplayName("core_gru")).toBe("AI ranking");
    expect(modelDisplayName("parametric_mlp")).toBe("AI ranking");
  });

  it("formats evidence levels", () => {
    expect(formatEvidenceLevel("MODERATE_EVIDENCE").label).toBe("MODERATE_EVIDENCE");
  });
});

describe("formatUnit", () => {
  it("formats values with units", () => {
    expect(formatUnit(25, "mV")).toBe("25 mV");
  });

  it("handles null", () => {
    expect(formatUnit(null, "mV")).toBe("—");
  });
});
