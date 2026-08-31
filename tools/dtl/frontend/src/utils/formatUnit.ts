export function formatUnit(value: number | null | undefined, unit: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const formatted = Number.isInteger(value) ? value.toString() : value.toFixed(4).replace(/\.?0+$/, "");
  return unit ? `${formatted} ${unit}` : formatted;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(2)}%`;
}

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(4);
}
