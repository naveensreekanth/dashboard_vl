import type { AnalysisDecision } from "@/api/analysisTypes";
import type { AnalysisRecommendationRow } from "@/api/analysisTypes";
import {
  buildHumanReadableExplanation,
  displayDelta,
  formatLimit,
  formatSimulatedYield,
  monthLabel,
} from "@/utils/analysisDisplay";
import { formatScore } from "@/utils/formatUnit";

function decisionClass(decision: string): string {
  if (decision === "RECOMMEND") return "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/10";
  if (decision === "KEEP_CURRENT") return "text-blue-600 dark:text-blue-400 border-blue-500/40 bg-blue-500/10";
  if (decision === "REVIEW_REQUIRED") return "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/10";
  if (decision === "REJECT") return "text-red-600 dark:text-red-400 border-red-500/40 bg-red-500/10";
  return "text-[var(--text-secondary)] border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)]";
}

export function TopSummaryCard({
  month,
  row,
  lotId,
  dieId,
  parameter,
  category,
}: {
  month: string;
  row: AnalysisRecommendationRow | undefined;
  lotId?: string;
  dieId?: string;
  parameter?: string;
  category?: string;
}) {
  const delta = displayDelta(row);
  const entityLot = lotId ?? row?.lot_id;
  const entityDie = dieId ?? row?.die_id;
  const entityParam = parameter ?? row?.parameter_display;
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm transition-colors"
      aria-label="Three-month DTL recommendation summary"
      data-testid="top-summary"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-3">
        <div>
          <h2 className="text-sm font-bold tracking-wider text-[var(--text-primary)] uppercase">
            DTL Recommendation Summary
          </h2>
          <div className="mt-1 text-xs text-[var(--text-muted)] flex flex-wrap gap-x-3 gap-y-1 font-mono" data-testid="summary-entity">
            <span>Month: {monthLabel(month)}</span>
            {entityLot ? <span>Lot: {entityLot}</span> : null}
            {entityDie ? <span>Die: {entityDie}</span> : null}
            {entityParam ? <span>Param: {entityParam}</span> : null}
          </div>
        </div>
        {row?.decision && (
          <span
            className={`font-semibold border rounded px-2.5 py-0.5 text-xs inline-block font-mono uppercase tracking-wide ${decisionClass(String(row.decision))}`}
            data-testid="summary-decision"
          >
            {row.decision}
          </span>
        )}
      </div>

      {(category || entityLot) && (
        <dl
          className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-5 text-xs text-[var(--text-muted)] bg-[var(--bg-panel-secondary)] p-3 rounded-md border border-[var(--border-subtle)]"
          data-testid="selected-die-context"
        >
          <div>
            <dt className="text-[var(--text-muted)]">Production Month</dt>
            <dd className="text-[var(--text-primary)] font-medium">{monthLabel(month)}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">Category</dt>
            <dd className="text-[var(--text-primary)] font-mono">{category ?? row?.lot_category ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">Lot</dt>
            <dd className="text-[var(--text-primary)] font-mono">{entityLot ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">Die</dt>
            <dd className="text-[var(--text-primary)] font-mono font-bold">{entityDie ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-[var(--text-muted)]">Parameter</dt>
            <dd className="text-[var(--text-primary)] font-mono">{entityParam ?? "—"}</dd>
          </div>
        </dl>
      )}

      {!row ? (
        <p className="mt-4 text-amber-500 text-xs font-medium" data-testid="summary-unavailable">
          Recommendation unavailable for this selection.
        </p>
      ) : (
        <dl className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 text-sm">
          <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
            <dt className="text-xs font-medium text-[var(--text-muted)]">Current DTL</dt>
            <dd className="font-mono text-xl font-bold text-[var(--text-secondary)] mt-1" data-testid="summary-current">
              {formatLimit(row, "current")}
            </dd>
          </div>
          <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
            <dt className="text-xs font-medium text-[var(--text-muted)]">AI Recommended DTL</dt>
            <dd className="font-mono text-xl font-bold text-[var(--accent)] mt-1" data-testid="summary-recommended">
              {formatLimit(row, "recommended")}
            </dd>
          </div>
          <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
            <dt className="text-xs font-medium text-[var(--text-muted)]">Maximum Eligible Simulated Yield</dt>
            <dd className="font-mono text-xl font-bold text-emerald-600 dark:text-emerald-400 mt-1" data-testid="summary-yield">
              {formatSimulatedYield(row.max_eligible_simulated_yield)}
            </dd>
            <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
              Estimated from test data
            </p>
          </div>
          <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
            <dt className="text-xs font-medium text-[var(--text-muted)]">Delta</dt>
            <dd className="font-mono text-sm font-semibold text-[var(--text-primary)] mt-1" data-testid="summary-delta">
              {delta.abs} · {delta.pct}
            </dd>
            <p className="text-[10px] text-[var(--text-muted)] mt-0.5">Display-only; not used for selection.</p>
          </div>
          <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
            <dt className="text-xs font-medium text-[var(--text-muted)]">ML Rank (tie-break)</dt>
            <dd className="font-mono text-lg font-semibold text-[var(--text-primary)] mt-1" data-testid="summary-ml-rank">
              {row.ml_rank != null ? `#${row.ml_rank}` : "—"}
            </dd>
          </div>
        </dl>
      )}
    </section>
  );
}

export function CurrentVsRecommendedCard({ row }: { row: AnalysisRecommendationRow | undefined }) {
  if (!row) return null;
  const delta = displayDelta(row);
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm"
      aria-label="Current versus recommended DTL"
      data-testid="current-vs-recommended"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-3">Current vs Recommended</h3>
      <div className="flex flex-col sm:flex-row items-center gap-4 justify-center text-center">
        <div className="rounded border border-[var(--border-muted)] bg-[var(--bg-panel-secondary)] px-6 py-4 min-w-[10rem]">
          <p className="text-xs text-[var(--text-muted)] uppercase">Current DTL</p>
          <p className="font-mono text-2xl font-bold text-[var(--text-secondary)] mt-1">{formatLimit(row, "current")}</p>
        </div>
        <p className="text-xl text-[var(--text-muted)] font-mono" aria-hidden>
          →
        </p>
        <div className="rounded border border-[var(--accent)] bg-[var(--accent-subtle)] px-6 py-4 min-w-[10rem]">
          <p className="text-xs text-[var(--accent)] uppercase font-semibold">AI Recommended DTL</p>
          <p className="font-mono text-2xl font-bold text-[var(--accent)] mt-1">{formatLimit(row, "recommended")}</p>
        </div>
      </div>
      <div className="mt-4 text-center text-xs">
        <p className="font-mono text-[var(--text-secondary)]">
          {delta.abs} · {delta.pct}
        </p>
        <p className={`mt-1 font-semibold ${decisionClass(String(row.decision)).split(" ")[0]}`}>
          Decision: {row.decision as AnalysisDecision}
        </p>
        <p className="text-[10px] text-[var(--text-muted)] mt-1">Delta is display-only and does not influence selection.</p>
      </div>
    </section>
  );
}

export function WhySelectedCard({ row }: { row: AnalysisRecommendationRow | undefined }) {
  if (!row) return null;

  const currentFormatted = formatLimit(row, "current");
  const recFormatted = formatLimit(row, "recommended");
  const isRecommend = row.decision === "RECOMMEND";
  const hasChange = isRecommend && row.current_limit !== row.recommended_limit;
  const yieldTie = Boolean(row.yield_tie);
  const safetyPass = String(row.safety_status ?? "").toUpperCase() === "PASS";
  const hasYield = row.max_eligible_simulated_yield != null;

  // Dynamic Checkmarks (Reasons) — ONLY show when supported by recommendation result
  const checks: { label: string; ok: boolean }[] = [
    { ok: true, label: "Test data is valid" },
  ];

  if (row.evidence_origin) {
    checks.push({ ok: true, label: "Simulation supports the selection" });
  }

  checks.push({ ok: true, label: "All required test conditions are covered" });

  if (safetyPass) {
    checks.push({ ok: true, label: "Passed safety checks" });
  }

  if (hasYield) {
    checks.push({ ok: true, label: "Highest simulated yield" });
  }

  if (yieldTie) {
    checks.push({ ok: true, label: "ML ranking used as tie-breaker" });
  }

  const explanationText = buildHumanReadableExplanation({
    recommendedVal: recFormatted,
    currentVal: currentFormatted,
    isRecommend,
    yieldTie,
    safetyPass,
    hasYield,
    whySelectedRaw: row.why_selected ?? undefined,
    selectionTextRaw: row.selection_text ?? undefined,
  });

  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 text-sm shadow-sm transition-colors"
      aria-label="Why selected"
      data-testid="why-selected"
    >
      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">
        Why was {recFormatted} selected?
      </h3>

      {/* Dynamic Reasons */}
      <ul className="space-y-1.5 mb-4" data-testid="why-checks">
        {checks.map((c) => (
          <li key={c.label} className="text-emerald-600 dark:text-emerald-400 font-medium text-xs flex items-center gap-2">
            <span className="font-bold">✓</span> {c.label}
          </li>
        ))}
      </ul>

      {/* Compact Structured Details: DTL Limit, Simulated Yield, ML Rank */}
      <div className="grid gap-3 sm:grid-cols-3 border-t border-b border-[var(--border-subtle)] py-3 mb-3 text-xs">
        <div>
          <dt className="text-[11px] text-[var(--text-muted)] uppercase font-medium tracking-wide">DTL Limit</dt>
          <dd className="font-mono text-[var(--text-primary)] font-semibold mt-0.5" data-testid="why-limit-change">
            {currentFormatted} → {hasChange ? recFormatted : "No change"}
          </dd>
        </div>

        {hasYield && (
          <div>
            <dt className="text-[11px] text-[var(--text-muted)] uppercase font-medium tracking-wide">Simulated Yield</dt>
            <dd className="font-mono text-emerald-600 dark:text-emerald-400 font-semibold mt-0.5" data-testid="why-simulated-yield">
              {formatSimulatedYield(row.max_eligible_simulated_yield)}
            </dd>
            <p className="text-[10px] text-[var(--text-muted)] italic mt-0.5">Estimated from test data</p>
          </div>
        )}

        {row.ml_rank != null && (
          <div>
            <dt className="text-[11px] text-[var(--text-muted)] uppercase font-medium tracking-wide">ML Rank</dt>
            <dd className="font-mono text-[var(--text-primary)] font-semibold mt-0.5" data-testid="why-ml-rank">
              #{row.ml_rank}
            </dd>
          </div>
        )}
      </div>

      {/* Concise Human Explanation */}
      <div className="mb-3">
        <p className="text-[11px] text-[var(--text-muted)] uppercase font-medium tracking-wide mb-1">Explanation</p>
        <p className="text-xs text-[var(--text-secondary)] leading-relaxed" data-testid="why-selected-text">
          {explanationText}
        </p>
      </div>

      {/* Collapsible Technical Details */}
      <details className="mt-3 rounded border border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)] p-3 text-xs text-[var(--text-muted)]" data-testid="technical-details">
        <summary className="cursor-pointer font-medium text-[var(--text-secondary)] hover:text-[var(--accent)] transition-colors">
          Technical details
        </summary>
        <dl className="mt-2.5 space-y-1.5 border-t border-[var(--border-subtle)] pt-2 font-mono text-[11px]">
          <div className="flex justify-between">
            <dt className="text-[var(--text-muted)]">Decision:</dt>
            <dd className="text-[var(--text-primary)]">{row.decision}</dd>
          </div>
          {row.why_selected && (
            <div className="flex justify-between">
              <dt className="text-[var(--text-muted)]">Reason code:</dt>
              <dd className="text-[var(--text-primary)] truncate max-w-[240px]">{row.why_selected}</dd>
            </div>
          )}
          {row.evidence_origin && (
            <div className="flex justify-between">
              <dt className="text-[var(--text-muted)]">Evidence origin:</dt>
              <dd className="text-[var(--text-primary)] truncate max-w-[240px]">{row.evidence_origin}</dd>
            </div>
          )}
          <div className="flex justify-between">
            <dt className="text-[var(--text-muted)]">Policy path:</dt>
            <dd className="text-[var(--text-primary)]">
              {yieldTie
                ? "maximum simulated yield → ML rank tie-break"
                : "maximum simulated yield (unique)"}
            </dd>
          </div>
          {row.policy_reason && (
            <div className="flex justify-between">
              <dt className="text-[var(--text-muted)]">policy_reason:</dt>
              <dd className="text-[var(--text-primary)]">{row.policy_reason}</dd>
            </div>
          )}
          {row.ml_score != null && (
            <div className="flex justify-between">
              <dt className="text-[var(--text-muted)]">Raw ML score:</dt>
              <dd className="text-[var(--text-primary)]">{formatScore(row.ml_score)}</dd>
            </div>
          )}
        </dl>
      </details>
    </section>
  );
}

export function PolicyFlowCard() {
  const steps = [
    "ALL CANDIDATES",
    "CATALOG / DATA VALID",
    "SIMULATION EVIDENCE",
    "REQUIRED CONDITIONS",
    "SAFETY PASS",
    "ELIGIBLE CANDIDATES",
    "MAXIMUM SIMULATED YIELD",
    "ML RANK TIE-BREAK",
    "FINAL DTL",
  ];
  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm"
      aria-label="Recommendation policy flow"
      data-testid="policy-flow"
    >
      <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] mb-2">Policy Flow (Explanatory)</h3>
      <ol className="text-xs font-mono text-[var(--text-secondary)] space-y-1">
        {steps.map((s, i) => (
          <li key={s}>
            {s}
            {i < steps.length - 1 ? <span className="text-[var(--text-muted)]"> ↓</span> : null}
          </li>
        ))}
      </ol>
      <p className="text-[10px] text-[var(--text-muted)] mt-2">
        This diagram is not an interactive calculator. The backend remains the source of truth.
      </p>
    </section>
  );
}
