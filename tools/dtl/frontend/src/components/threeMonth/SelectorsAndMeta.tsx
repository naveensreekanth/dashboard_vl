import type {
  AnalysisRecommendationRow,
  ModelTraceRow,
  SameDieRow,
} from "@/api/analysisTypes";
import { MONTH_OPTIONS } from "@/api/analysisTypes";
import { formatSimulatedYield, shortMonth } from "@/utils/analysisDisplay";
import { formatUnit } from "@/utils/formatUnit";
import { AdvancedEvidence } from "@/components/common/AdvancedEvidence";

export function MonthSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (month: string) => void;
}) {
  return (
    <fieldset className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-3.5 shadow-sm transition-colors" data-testid="month-selector">
      <legend className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] px-1">Production Month</legend>
      <div className="flex flex-wrap gap-2 mt-1">
        {MONTH_OPTIONS.map((m) => {
          const isActive = value === m.value;
          return (
            <button
              key={m.value}
              type="button"
              onClick={() => onChange(m.value)}
              className={`rounded px-3 py-1.5 text-xs font-medium border transition-colors ${
                isActive
                  ? "border-[var(--accent)] bg-[var(--accent-subtle)] text-[var(--accent)] font-semibold"
                  : "border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)] text-[var(--text-secondary)] hover:border-[var(--border-muted)] hover:text-[var(--text-primary)]"
              }`}
              data-testid={`month-${m.value}`}
              aria-pressed={isActive}
            >
              {m.label}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

export function ParameterSelector(props: {
  value: string;
  options: string[];
  onChange: (p: string) => void;
  nonScorable?: string[];
  nonScorableNote?: string;
}) {
  const { value, options, onChange } = props;
  return (
    <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-3.5 shadow-sm transition-colors" data-testid="parameter-selector">
      <label className="block text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
        Parameter
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="mt-1.5 w-full rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] px-3 py-2 text-xs font-mono text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          data-testid="parameter-select"
        >
          {options.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

export function ExecutiveMatrix({
  primaryRows,
  parameters,
  selectedParameter,
  onSelectParameter,
}: {
  primaryRows: AnalysisRecommendationRow[];
  parameters: string[];
  selectedParameter?: string;
  onSelectParameter: (p: string) => void;
}) {
  const months = ["2026-01", "2026-02", "2026-03"] as const;
  const monthHeaders = ["January 2026", "February 2026", "March 2026"];
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm overflow-x-auto transition-colors"
      data-testid="executive-matrix"
      aria-label="AI recommended Dynamic Test Limits across three months"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-3 mb-3">
        <div>
          <h2 className="text-sm font-bold tracking-wider text-[var(--text-primary)] uppercase">
            AI RECOMMENDED DYNAMIC TEST LIMITS
          </h2>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            Recommended Dynamic Test Limits across January, February, and March 2026.
          </p>
        </div>
        <span className="text-[11px] font-mono text-[var(--text-muted)]">
          9 Parameters · 3 Months
        </span>
      </div>

      <table className="w-full text-xs text-left border-collapse">
        <thead>
          <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            <th className="py-2.5 pr-4">Parameter</th>
            {monthHeaders.map((label) => (
              <th key={label} className="py-2.5 pr-4 font-semibold text-[var(--text-secondary)]">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border-subtle)]">
          {parameters.map((p) => {
            const cells = months.map((m) =>
              primaryRows.find((r) => r.parameter_display === p && r.production_month === m),
            );
            const limits = cells.map((r) =>
              r?.recommended_limit != null ? Number(r.recommended_limit) : null,
            );
            const defined = limits.filter((v): v is number => v != null);
            const changesAcrossMonths =
              defined.length >= 2 && new Set(defined.map((v) => v.toFixed(12))).size > 1;
            const isSelected = selectedParameter === p;
            return (
              <tr
                key={p}
                className={`transition-colors ${
                  isSelected
                    ? "bg-[var(--accent-subtle)] font-medium"
                    : "hover:bg-[var(--bg-panel-secondary)]"
                }`}
              >
                <td className="py-2.5 pr-4">
                  <button
                    type="button"
                    className={`font-mono text-left text-xs cursor-pointer rounded px-1.5 py-0.5 -mx-1 transition-colors hover:underline focus:outline-none focus:ring-1 focus:ring-[var(--accent)] ${
                      isSelected
                        ? "text-[var(--accent)] font-bold"
                        : "text-[var(--text-primary)] hover:text-[var(--accent)]"
                    }`}
                    onClick={() => onSelectParameter(p)}
                    aria-pressed={isSelected}
                    aria-label={`Inspect engineering detail for ${p}`}
                    data-testid={`matrix-param-${p}`}
                  >
                    {p}
                  </button>
                </td>
                {cells.map((r, i) => {
                  const text = r ? formatUnit(r.recommended_limit, r.unit ?? "") : "—";
                  const valueClass = changesAcrossMonths
                    ? "text-[var(--accent)] font-semibold"
                    : "text-[var(--text-secondary)]";
                  return (
                    <td
                      key={months[i]}
                      className="py-2.5 pr-4"
                      data-changed={changesAcrossMonths || undefined}
                    >
                      <button
                        type="button"
                        className={`font-mono text-left text-xs cursor-pointer rounded px-1.5 py-0.5 -mx-1 transition-colors hover:underline focus:outline-none focus:ring-1 focus:ring-[var(--accent)] ${valueClass}`}
                        onClick={() => onSelectParameter(p)}
                        aria-label={`Inspect engineering detail for ${p} (${monthHeaders[i]})`}
                        data-testid={`matrix-value-${p}-${months[i]}`}
                      >
                        {text}
                      </button>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-[11px] text-[var(--text-muted)] mt-3">
        Highlighted values indicate parameters whose recommended DTL changes across months. Click a
        parameter to inspect engineering detail below.
      </p>
    </section>
  );
}

export function SameDiePanel({
  rows,
  parameterDisplay,
  lotId,
  dieId,
  category,
  lotOptions,
  onLotDieChange,
}: {
  rows: SameDieRow[];
  parameterDisplay: string;
  lotId: string;
  dieId: string;
  category: string;
  lotOptions: Array<{ lot_id: string; die_id: string; lot_category: string }>;
  onLotDieChange: (lot: string, die: string) => void;
}) {
  const months = ["2026-01", "2026-02", "2026-03"];
  const byMonth = Object.fromEntries(
    months.map((m) => [
      m,
      rows.find(
        (r) =>
          r.production_month === m &&
          r.parameter_display === parameterDisplay &&
          r.lot_id === lotId &&
          r.die_id === dieId,
      ),
    ]),
  );
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm overflow-x-auto transition-colors"
      data-testid="same-die-panel"
    >
      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-2">Same Die — Three-Month Response</h3>
      <label className="block text-xs text-[var(--text-muted)] mb-3">
        Lot / Die
        <select
          className="mt-1 w-full rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] px-3 py-2 text-xs font-mono text-[var(--text-primary)]"
          value={`${lotId}::${dieId}`}
          onChange={(e) => {
            const parts = e.target.value.split("::");
            const l = parts[0] ?? "";
            const d = parts[1] ?? "";
            onLotDieChange(l, d);
          }}
          data-testid="same-die-select"
        >
          {lotOptions.map((o) => (
            <option key={`${o.lot_id}::${o.die_id}`} value={`${o.lot_id}::${o.die_id}`}>
              {o.lot_category}: {o.lot_id} / {o.die_id}
            </option>
          ))}
        </select>
      </label>
      <p className="text-xs text-[var(--text-muted)] mb-2 font-mono">
        Category: {category} · Parameter: {parameterDisplay}
      </p>
      <table className="w-full text-xs text-left border-collapse">
        <thead>
          <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            <th className="py-2">Metric</th>
            {months.map((m) => (
              <th key={m} className="py-2 font-semibold text-[var(--text-secondary)]">
                {shortMonth(m)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border-subtle)]">
          <tr>
            <td className="py-2 text-[var(--text-secondary)]">Observed mean</td>
            {months.map((m) => (
              <td key={m} className="py-2 font-mono text-[var(--text-primary)]">
                {byMonth[m]?.observed_mean != null
                  ? Number(byMonth[m]?.observed_mean).toFixed(3)
                  : "—"}
              </td>
            ))}
          </tr>
          <tr>
            <td className="py-2 text-[var(--text-secondary)]">Recommended DTL</td>
            {months.map((m) => (
              <td key={m} className="py-2 font-mono text-[var(--accent)] font-semibold">
                {byMonth[m]?.recommended_limit ?? "—"}
              </td>
            ))}
          </tr>
          <tr>
            <td className="py-2 text-[var(--text-secondary)]">Simulated Yield</td>
            {months.map((m) => (
              <td key={m} className="py-2 font-mono text-[var(--text-primary)]">
                {formatSimulatedYield(byMonth[m]?.max_eligible_simulated_yield)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="py-2 text-[var(--text-secondary)]">ML Rank</td>
            {months.map((m) => (
              <td key={m} className="py-2 font-mono text-[var(--text-primary)]">
                {byMonth[m]?.ml_rank != null ? `#${byMonth[m]?.ml_rank}` : "—"}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </section>
  );
}

export function ModelTraceTable({ rows }: { rows: ModelTraceRow[] }) {
  void rows;
  return null;
}

export function AdvancedAnalysisEvidence({
  row,
  candidatesCount,
}: {
  row: AnalysisRecommendationRow | undefined;
  candidatesCount: number;
}) {
  if (!row) return null;
  return (
    <AdvancedEvidence>
      <section className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-4" data-testid="advanced-analysis">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-3">Advanced Evidence</h3>
        <dl className="grid sm:grid-cols-2 gap-2 text-xs font-mono text-[var(--text-secondary)]">
          <div>
            <dt className="text-[var(--text-muted)]">production_month</dt>
            <dd className="text-[var(--text-primary)]">{row.production_month}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">lot_id / die_id</dt>
            <dd className="text-[var(--text-primary)]">
              {row.lot_id} / {row.die_id}
            </dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">sequence_id</dt>
            <dd className="text-[var(--text-primary)]">{row.sequence_id ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">ml_score / ml_rank</dt>
            <dd className="text-[var(--text-primary)]">
              {row.ml_score ?? "—"} / {row.ml_rank ?? "—"}
            </dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">current / recommended</dt>
            <dd className="text-[var(--text-primary)]">
              {row.current_limit} → {row.recommended_limit}
            </dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">recommendation_delta</dt>
            <dd className="text-[var(--text-primary)]">{row.recommendation_delta ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">max_eligible_simulated_yield</dt>
            <dd className="text-[var(--text-primary)]">{row.max_eligible_simulated_yield ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">safety_status</dt>
            <dd className="text-[var(--text-primary)]">{row.safety_status ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">policy_reason</dt>
            <dd className="text-[var(--text-primary)]">{row.policy_reason ?? "—"}</dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-[var(--text-muted)]">selection_text</dt>
            <dd className="whitespace-pre-wrap text-[var(--text-primary)]">{row.selection_text ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">gate-set candidates</dt>
            <dd className="text-[var(--text-primary)]">{candidatesCount}</dd>
          </div>
        </dl>
      </section>
    </AdvancedEvidence>
  );
}
