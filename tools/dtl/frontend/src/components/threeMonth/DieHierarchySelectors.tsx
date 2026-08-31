import type { DieLevelIdentities } from "@/api/analysisTypes";

const CATEGORIES = ["NORMAL", "SCRATCH", "EDGE", "CENTER"] as const;

export function DieHierarchySelectors({
  identities,
  category,
  lotId,
  dieId,
  parameter,
  parameters,
  onCategoryChange,
  onLotChange,
  onDieChange,
  onParameterChange,
}: {
  identities: DieLevelIdentities | null | undefined;
  category: string;
  lotId: string;
  dieId: string;
  parameter: string;
  parameters: string[];
  onCategoryChange: (c: string) => void;
  onLotChange: (lot: string) => void;
  onDieChange: (die: string) => void;
  onParameterChange: (p: string) => void;
}) {
  const lots = identities?.lots_by_category?.[category] ?? [];
  const dies = identities?.dies_by_lot?.[lotId] ?? [];

  return (
    <div
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-3.5 shadow-sm transition-colors"
      data-testid="die-hierarchy-selectors"
    >
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <label className="block text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Lot Category
          <select
            value={category}
            onChange={(e) => onCategoryChange(e.target.value)}
            className="mt-1.5 w-full rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] px-2.5 py-1.5 text-xs font-mono text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            data-testid="category-select"
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Lot
          <select
            value={lotId}
            onChange={(e) => onLotChange(e.target.value)}
            className="mt-1.5 w-full rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] px-2.5 py-1.5 text-xs font-mono text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            data-testid="lot-select"
          >
            {lots.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Die
          <select
            value={dieId}
            onChange={(e) => onDieChange(e.target.value)}
            className="mt-1.5 w-full rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] px-2.5 py-1.5 text-xs font-mono text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            data-testid="die-select"
          >
            {dies.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Parameter
          <select
            value={parameter}
            onChange={(e) => onParameterChange(e.target.value)}
            className="mt-1.5 w-full rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] px-2.5 py-1.5 text-xs font-mono text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            data-testid="hierarchy-parameter-select"
          >
            {parameters.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
      </div>
      {identities?.note ? (
        <p className="mt-2 text-[10px] text-[var(--text-muted)] font-mono">
          Lot and die identities are stable across January–March 2026.
        </p>
      ) : null}
    </div>
  );
}
