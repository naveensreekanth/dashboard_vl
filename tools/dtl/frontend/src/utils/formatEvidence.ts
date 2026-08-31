import type { EvidenceLevel } from "@/api/types";

export interface EvidenceLevelDisplay {
  label: string;
  className: string;
}

const LEVEL_MAP: Record<EvidenceLevel, EvidenceLevelDisplay> = {
  HIGH_EVIDENCE: { label: "HIGH_EVIDENCE", className: "text-green-400 border-green-700" },
  MODERATE_EVIDENCE: {
    label: "MODERATE_EVIDENCE",
    className: "text-cyan-400 border-cyan-700",
  },
  LOW_EVIDENCE: { label: "LOW_EVIDENCE", className: "text-amber-400 border-amber-700" },
  INSUFFICIENT_EVIDENCE: {
    label: "INSUFFICIENT_EVIDENCE",
    className: "text-gray-400 border-gray-600",
  },
};

export function formatEvidenceLevel(level: EvidenceLevel): EvidenceLevelDisplay {
  return LEVEL_MAP[level];
}

export function modelDisplayName(modelId: string | null | undefined): string {
  // Presentation only — never show architecture names (e.g. GRU) in the UI.
  if (!modelId) return "None";
  return "AI ranking";
}

export function isCoreParameter(parameter: string): boolean {
  return parameter === "ir_drop" || parameter === "thermal";
}
