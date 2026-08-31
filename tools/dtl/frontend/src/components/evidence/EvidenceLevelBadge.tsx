import type { EvidenceLevel } from "@/api/types";
import { formatEvidenceLevel } from "@/utils/formatEvidence";

export function EvidenceLevelBadge({ level }: { level: EvidenceLevel }) {
  const display = formatEvidenceLevel(level);
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium ${display.className}`}
      aria-label={`Evidence level ${display.label}`}
    >
      {display.label}
    </span>
  );
}
