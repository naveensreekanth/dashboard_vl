export function formatObservedRule(
  rule: string | undefined,
  opts?: { n?: number | null; conditionId?: string },
): string | null {
  if (!rule) return null;
  if (rule === "median_over_patterns") {
    if (opts?.n && opts.n > 0) {
      return `Median across ${opts.n} patterns`;
    }
    return "Median across patterns";
  }
  if (rule === "selected_condition") {
    if (opts?.conditionId) {
      return `Selected condition ${opts.conditionId}`;
    }
    return "Selected condition reading";
  }
  return rule;
}

export function isSyntheticSource(classification: string | undefined): boolean {
  return (classification ?? "").toUpperCase() === "SYNTHETIC";
}
