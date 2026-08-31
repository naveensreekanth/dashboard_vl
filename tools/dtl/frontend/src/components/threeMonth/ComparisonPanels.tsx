import type {
  AnalysisCandidateRow,
  AnalysisRecommendationRow,
  MlTieBreakProof,
  TemporalChangeRow,
  YieldFirstProof,
} from "@/api/analysisTypes";
import {
  formatSimulatedYield,
  shortMonth,
} from "@/utils/analysisDisplay";
import { formatScore, formatUnit } from "@/utils/formatUnit";

export function RecommendedTrendChart({
  history,
}: {
  history: AnalysisRecommendationRow[];
}) {
  if (history.length === 0) return null;
  const unit = history[0]?.unit ?? "";
  const current = history[0]?.current_limit ?? 0;
  const values = history.map((r) => r.recommended_limit);
  const min = Math.min(...values, current);
  const max = Math.max(...values, current);
  const span = max - min || 1;
  const w = 360;
  const h = 160;
  const pad = 28;
  const points = history.map((r, i) => {
    const x = pad + (i * (w - 2 * pad)) / Math.max(history.length - 1, 1);
    const y = h - pad - ((r.recommended_limit - min) / span) * (h - 2 * pad);
    return { x, y, label: shortMonth(r.production_month), v: r.recommended_limit };
  });
  const currentY = h - pad - ((current - min) / span) * (h - 2 * pad);
  const polyline = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm transition-colors"
      aria-label="Recommended DTL by production month"
      data-testid="recommended-trend-chart"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-1">Recommended DTL by Production Month</h3>
      <p className="text-[11px] text-[var(--text-muted)] mb-3">
        Rising/falling DTL does not automatically mean better production yield.
      </p>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-40" role="img">
        <title>Recommended DTL trend</title>
        {/* Current baseline reference */}
        <line
          x1={pad}
          x2={w - pad}
          y1={currentY}
          y2={currentY}
          stroke="#94a3b8"
          strokeDasharray="4 4"
          strokeWidth="1.5"
        />
        {/* Recommended polyline */}
        <polyline fill="none" stroke="#0891b2" strokeWidth="2.5" points={polyline} className="dark:stroke-cyan-400" />
        {points.map((p) => (
          <g key={p.label}>
            <circle cx={p.x} cy={p.y} r="4.5" fill="#0891b2" className="dark:fill-cyan-400" />
            <text x={p.x} y={h - 6} textAnchor="middle" fill="currentColor" className="text-[10px] font-mono text-[var(--text-muted)]">
              {p.label}
            </text>
            <text x={p.x} y={p.y - 8} textAnchor="middle" fill="currentColor" className="text-[10px] font-mono font-bold text-[var(--text-primary)]">
              {p.v}
            </text>
          </g>
        ))}
      </svg>
      <p className="text-[11px] text-[var(--text-muted)] mt-1 font-mono">
        Cyan: Recommended · Gray dotted: Current DTL ({formatUnit(current, unit)})
      </p>
    </section>
  );
}

export function YieldTrendTable({ history }: { history: AnalysisRecommendationRow[] }) {
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm transition-colors"
      aria-label="Maximum eligible simulated yield by month"
      data-testid="yield-trend"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-2">Maximum Eligible Simulated Yield</h3>
      <table className="w-full text-xs text-left border-collapse">
        <thead>
          <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            <th className="py-2.5">Month</th>
            <th className="py-2.5">Max Eligible Simulated Yield</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border-subtle)]">
          {history.map((r) => (
            <tr key={r.production_month} className="hover:bg-[var(--bg-panel-secondary)]">
              <td className="py-2.5 text-[var(--text-secondary)]">{shortMonth(r.production_month)}</td>
              <td className="py-2.5 font-mono text-emerald-600 dark:text-emerald-400 font-semibold">{formatSimulatedYield(r.max_eligible_simulated_yield)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function CandidateComparisonTable({
  candidates,
}: {
  candidates: AnalysisCandidateRow[];
}) {
  const gated = candidates
    .filter((c) => c.in_policy_gate_set !== false)
    .slice()
    .sort((a, b) => Number(a.ml_rank ?? 999) - Number(b.ml_rank ?? 999));
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm overflow-x-auto transition-colors"
      aria-label="Candidate comparison"
      data-testid="candidate-comparison"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-1">Candidate Comparison</h3>
      <p className="text-[11px] text-[var(--text-muted)] mb-3">
        Values from the analysis artifact. Evaluated candidate grid with simulation and ML ranking.
      </p>
      <table className="w-full text-xs text-left border-collapse">
        <thead>
          <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            <th className="py-2.5 pr-3">Candidate</th>
            <th className="py-2.5 pr-3">Simulated Yield</th>
            <th className="py-2.5 pr-3">Safety</th>
            <th className="py-2.5 pr-3">Eligible</th>
            <th className="py-2.5 pr-3">ML Score</th>
            <th className="py-2.5">ML Rank</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border-subtle)]">
          {gated.map((c) => (
            <tr
              key={`${c.candidate_limit}-${c.ml_rank}`}
              className={
                c.is_selected
                  ? "bg-[var(--accent-subtle)] font-medium text-[var(--accent)]"
                  : "hover:bg-[var(--bg-panel-secondary)] text-[var(--text-secondary)]"
              }
              data-selected={c.is_selected ? "true" : "false"}
            >
              <td className="py-2.5 pr-3 font-mono">
                {c.candidate_limit}
                {c.is_selected ? " ★" : ""}
                {c.is_current ? " (current)" : ""}
              </td>
              <td className="py-2.5 pr-3 font-mono">{formatSimulatedYield(c.simulated_yield)}</td>
              <td className="py-2.5 pr-3 font-mono">{c.safety_status ?? "—"}</td>
              <td className="py-2.5 pr-3">{c.eligible ? "Yes" : "No"}</td>
              <td className="py-2.5 pr-3 font-mono">{formatScore(c.ml_score)}</td>
              <td className="py-2.5 font-mono">{c.ml_rank != null ? `#${c.ml_rank}` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function YieldFirstInsightCard({ proof }: { proof: YieldFirstProof | null | undefined }) {
  if (!proof) return null;
  return (
    <section
      className="rounded-lg border border-emerald-500/30 bg-[var(--bg-panel)] p-5 shadow-sm"
      data-testid="yield-first-card"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-2">Yield-First Proof</h3>
      <p className="text-xs text-[var(--text-primary)] mb-2">Yield takes priority over ML rank.</p>
      <p className="text-[11px] font-mono text-[var(--text-muted)] mb-3">
        {proof.parameter_display} · {proof.production_month} · {proof.die_id}
      </p>
      <div className="grid sm:grid-cols-2 gap-3 text-xs">
        <div className="rounded border border-emerald-500/40 bg-emerald-500/5 dark:bg-emerald-950/20 p-3">
          <p className="text-[11px] text-[var(--text-muted)] uppercase">Winner (Selected)</p>
          <p className="font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">{proof.winner.candidate_limit}</p>
          <p className="font-mono text-[11px] mt-1 text-[var(--text-secondary)]">
            Yield {formatSimulatedYield(proof.winner.simulated_yield)} · ML Rank #
            {proof.winner.ml_rank}
          </p>
        </div>
        <div className="rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] p-3">
          <p className="text-[11px] text-[var(--text-muted)] uppercase">Higher ML, Lower Yield</p>
          <p className="font-mono text-sm font-bold text-[var(--text-primary)] mt-0.5">{proof.loser_higher_ml.candidate_limit}</p>
          <p className="font-mono text-[11px] mt-1 text-[var(--text-secondary)]">
            Yield {formatSimulatedYield(proof.loser_higher_ml.simulated_yield)} · ML Rank #
            {proof.loser_higher_ml.ml_rank}
          </p>
        </div>
      </div>
      {proof.statement ? <p className="text-[11px] text-[var(--text-muted)] mt-2 italic">{proof.statement}</p> : null}
    </section>
  );
}

export function MlTieBreakInsightCard({
  row,
  proof,
}: {
  row: AnalysisRecommendationRow | undefined;
  proof: MlTieBreakProof | null | undefined;
}) {
  const show =
    Boolean(row?.yield_tie) ||
    (proof &&
      row &&
      proof.parameter_display === row.parameter_display &&
      proof.production_month === row.production_month);
  if (!show) return null;
  const tied = proof?.tied_candidates ?? [];
  return (
    <section
      className="rounded-lg border border-indigo-500/30 bg-[var(--bg-panel)] p-5 shadow-sm"
      data-testid="ml-tie-break-card"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 mb-2">ML Tie-Break</h3>
      <p className="text-xs text-[var(--text-primary)]">
        Multiple eligible candidates achieved the same maximum simulated yield. ML rank was used as
        the tie-breaker.
      </p>
      {tied.length > 0 ? (
        <table className="w-full text-xs text-left border-collapse mt-3">
          <thead>
            <tr className="border-b border-[var(--border-subtle)] text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              <th className="py-2">Candidate</th>
              <th className="py-2">Yield</th>
              <th className="py-2">ML Score</th>
              <th className="py-2">ML Rank</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)]">
            {tied.map((t) => (
              <tr key={t.candidate_limit} className={t.is_selected ? "text-[var(--accent)] font-semibold" : "text-[var(--text-secondary)]"}>
                <td className="py-2 font-mono">
                  {t.candidate_limit}
                  {t.is_selected ? " ★" : ""}
                </td>
                <td className="py-2 font-mono">{formatSimulatedYield(t.simulated_yield)}</td>
                <td className="py-2 font-mono">{formatScore(t.ml_score)}</td>
                <td className="py-2 font-mono">#{t.ml_rank}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  );
}

export function MonthChangeCard({
  change,
  history,
}: {
  change: TemporalChangeRow | undefined;
  history: AnalysisRecommendationRow[];
}) {
  if (!change?.recommendation_changed) return null;
  return (
    <section
      className="rounded-lg border border-amber-500/30 bg-[var(--bg-panel)] p-5 shadow-sm"
      data-testid="month-change-card"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400 mb-2">Recommendation Changed</h3>
      <ul className="text-xs font-mono space-y-1 text-[var(--text-primary)]">
        {history.map((r) => (
          <li key={r.production_month}>
            {shortMonth(r.production_month)}: {formatUnit(r.recommended_limit, r.unit ?? "")}
          </li>
        ))}
      </ul>
      <p className="text-xs text-[var(--text-muted)] mt-3">
        Recommendation changed across production months based on month-specific production
        sequences, simulation evidence, and ML ranking.
      </p>
    </section>
  );
}

export function StableParamsCard({ stable }: { stable: string[] }) {
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm"
      data-testid="stable-params"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-2">Stable Across Three Months</h3>
      <p className="text-[10px] text-[var(--text-muted)] mb-2">
        Primary-die analysis stability across 3 months.
      </p>
      <ul className="text-xs font-mono text-[var(--text-secondary)] grid sm:grid-cols-2 gap-1">
        {stable.map((p) => (
          <li key={p}>{p}</li>
        ))}
      </ul>
    </section>
  );
}
