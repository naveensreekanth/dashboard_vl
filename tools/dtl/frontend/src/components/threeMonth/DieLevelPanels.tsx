import type {
  AnalysisRecommendationRow,
  DieLevelIdentities,
  ObservedSummaryPayload,
} from "@/api/analysisTypes";
import { formatSimulatedYield, monthLabel, shortMonth } from "@/utils/analysisDisplay";
import { formatUnit } from "@/utils/formatUnit";

export function SameDieThreeMonthHistory({
  dieId,
  history,
  loading,
}: {
  dieId: string;
  history: AnalysisRecommendationRow[];
  loading?: boolean;
}) {
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm overflow-x-auto transition-colors"
      data-testid="same-die-three-month-history"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-3 mb-3">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Same Die — Three-Month History
          </h3>
          <p className="text-xs text-[var(--text-muted)] mt-0.5 font-mono">
            Selected die: <span className="font-bold text-[var(--accent)]">{dieId}</span>
          </p>
        </div>
        <span className="text-[11px] font-mono text-[var(--text-muted)]">
          {history.length} Months Evaluated
        </span>
      </div>

      {loading ? (
        <p className="text-xs text-[var(--text-muted)]">Loading history…</p>
      ) : (
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              <th className="py-2.5 pr-3">Month</th>
              <th className="py-2.5 pr-3">Current DTL</th>
              <th className="py-2.5 pr-3">Recommended DTL</th>
              <th className="py-2.5 pr-3">Max Eligible Simulated Yield</th>
              <th className="py-2.5 pr-3">ML Rank</th>
              <th className="py-2.5">Decision</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)]">
            {history.map((r) => (
              <tr key={r.production_month} className="hover:bg-[var(--bg-panel-secondary)]">
                <td className="py-2.5 pr-3 font-medium text-[var(--text-secondary)]">{shortMonth(r.production_month)}</td>
                <td className="py-2.5 pr-3 font-mono text-[var(--text-secondary)]">
                  {formatUnit(r.current_limit, r.unit ?? "")}
                </td>
                <td className="py-2.5 pr-3 font-mono text-[var(--accent)] font-semibold">
                  {formatUnit(r.recommended_limit, r.unit ?? "")}
                </td>
                <td className="py-2.5 pr-3 font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
                  {formatSimulatedYield(r.max_eligible_simulated_yield)}
                </td>
                <td className="py-2.5 pr-3 font-mono text-[var(--text-primary)]">
                  {r.ml_rank != null ? `#${r.ml_rank}` : "—"}
                </td>
                <td className="py-2.5 font-mono text-[var(--text-primary)]">{r.decision}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export function ObservedDieSummary({
  payload,
}: {
  payload: ObservedSummaryPayload | null;
}) {
  if (!payload) return null;
  const months = ["2026-01", "2026-02", "2026-03"];
  const params = Object.keys(payload.observed_means);
  if (params.length === 0) return null;
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm overflow-x-auto transition-colors"
      data-testid="observed-die-summary"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-1">Observed Data (Selected Die)</h3>
      <p className="text-[11px] text-[var(--text-muted)] mb-3">
        Context only — not used to recalculate recommendations.
      </p>
      <table className="w-full text-xs text-left border-collapse">
        <thead>
          <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            <th className="py-2.5 pr-3">Parameter</th>
            {months.map((m) => (
              <th key={m} className="py-2.5 pr-3 font-semibold text-[var(--text-secondary)]">
                {shortMonth(m)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border-subtle)]">
          {params.map((p) => (
            <tr key={p} className="hover:bg-[var(--bg-panel-secondary)]">
              <td className="py-2.5 pr-3 font-mono text-[var(--text-primary)] font-medium">{p}</td>
              {months.map((m) => {
                const v = payload.observed_means[p]?.[m];
                return (
                  <td key={m} className="py-2.5 pr-3 font-mono text-[var(--text-secondary)]">
                    {v != null ? Number(v).toFixed(3) : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function CategorySummaryPanel({
  open,
  onToggle,
  identities,
}: {
  open: boolean;
  onToggle: () => void;
  identities: DieLevelIdentities | null | undefined;
}) {
  const cats = identities?.categories ?? ["NORMAL", "SCRATCH", "EDGE", "CENTER"];
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm transition-colors"
      data-testid="category-summary"
    >
      <button
        type="button"
        className="text-xs font-bold uppercase tracking-wider text-[var(--accent)] hover:underline flex items-center gap-1.5"
        onClick={onToggle}
        data-testid="category-summary-toggle"
      >
        <span>{open ? "▾" : "▸"}</span> Category Summary
      </button>
      {open && identities ? (
        <table className="w-full text-xs text-left border-collapse mt-3">
          <thead>
            <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              <th className="py-2.5">Category</th>
              <th className="py-2.5">Lots</th>
              <th className="py-2.5">Dies</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)]">
            {cats.map((c) => (
              <tr key={c} className="hover:bg-[var(--bg-panel-secondary)]">
                <td className="py-2.5 font-mono text-[var(--text-primary)]">{c}</td>
                <td className="py-2.5 font-mono text-[var(--text-secondary)]">{identities.lots_by_category?.[c]?.length ?? 0}</td>
                <td className="py-2.5 font-mono text-[var(--text-secondary)]">
                  {identities.counts?.by_category_dies?.[c] ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {open ? (
        <p className="text-[10px] text-[var(--text-muted)] mt-2 font-mono">
          Recommendation distributions appear in Lot Summary after engine-backed browse for a lot.
        </p>
      ) : null}
    </section>
  );
}

export function PopulationViewPanel({
  open,
  onToggle,
  identities,
}: {
  open: boolean;
  onToggle: () => void;
  identities: DieLevelIdentities | null | undefined;
}) {
  const months = identities?.months ?? ["2026-01", "2026-02", "2026-03"];
  const lots = identities?.counts?.lots ?? 20;
  const dies = identities?.counts?.dies ?? 1000;
  const cov = identities?.cache_coverage;
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm transition-colors"
      data-testid="population-view"
    >
      <button
        type="button"
        className="text-xs font-bold uppercase tracking-wider text-[var(--accent)] hover:underline flex items-center gap-1.5"
        onClick={onToggle}
        data-testid="population-view-toggle"
      >
        <span>{open ? "▾" : "▸"}</span> Three-Month Population View
      </button>
      {open ? (
        <>
          <table className="w-full text-xs text-left border-collapse mt-3">
            <thead>
              <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                <th className="py-2.5">Month</th>
                <th className="py-2.5">Lots</th>
                <th className="py-2.5">Dies</th>
                <th className="py-2.5">Cached Recommendations</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)]">
              {months.map((m) => (
                <tr key={m} className="hover:bg-[var(--bg-panel-secondary)]">
                  <td className="py-2.5 text-[var(--text-primary)]">{monthLabel(m)}</td>
                  <td className="py-2.5 font-mono text-[var(--text-secondary)]">{lots}</td>
                  <td className="py-2.5 font-mono text-[var(--text-secondary)]">{dies}</td>
                  <td className="py-2.5 font-mono text-[var(--accent)]">
                    {cov?.by_month?.[m] != null ? cov.by_month[m] : "on-demand"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-[10px] text-[var(--text-muted)] mt-2 font-mono">
            Values show disk cache coverage only ({cov?.cached_files ?? 0} files total).
          </p>
        </>
      ) : null}
    </section>
  );
}
