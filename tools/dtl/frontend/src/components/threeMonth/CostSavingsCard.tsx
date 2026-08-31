import type { CostSavingsPayload } from "@/api/analysisTypes";

function money(n: number): { primary: string; centsSub?: string } {
  if (!Number.isFinite(n)) return { primary: "—" };
  const abs = Math.abs(n);
  if (abs === 0) return { primary: "$0.0000" };
  if (abs >= 1) return { primary: `$${n.toFixed(2)}` };
  if (abs >= 0.01) return { primary: `$${n.toFixed(4)}` };
  // Small fractional dollar amount: format cleanly without scientific notation
  const formattedDecimal = `$${n.toFixed(6).replace(/\.?0+$/, "")}`;
  const centsVal = (n * 100).toFixed(3).replace(/\.?0+$/, "");
  return {
    primary: formattedDecimal,
    centsSub: `${centsVal}¢ / die`,
  };
}

function seconds(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return `${n.toFixed(3)} s`;
}

export function PredictedCostSavingsCard({
  payload,
  loading,
  error,
}: {
  payload: CostSavingsPayload | null;
  loading?: boolean;
  error?: string | null;
}) {
  if (loading) {
    return (
      <section
        className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5"
        data-testid="cost-savings-card"
      >
        <h3 className="text-xs font-semibold tracking-wide uppercase text-[var(--text-muted)]">
          Predicted Cost Saving
        </h3>
        <p className="mt-2 text-xs text-[var(--text-secondary)]">Loading estimated cost saving…</p>
      </section>
    );
  }

  if (error) {
    return (
      <section
        className="rounded-lg border border-amber-500/30 bg-[var(--bg-panel)] p-5"
        data-testid="cost-savings-card"
      >
        <h3 className="text-xs font-semibold tracking-wide uppercase text-[var(--text-muted)]">
          Predicted Cost Saving
        </h3>
        <p className="mt-2 text-xs text-amber-500" data-testid="cost-savings-error">
          {error}
        </p>
      </section>
    );
  }

  if (!payload) return null;

  const agg = payload.aggregate;
  const scope = payload.selected_scope;
  const moneyFormatted = money(agg.total_predicted_cost_saving);

  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm transition-colors"
      aria-label="Predicted DTL test-time cost saving"
      data-testid="cost-savings-card"
    >
      <div className="flex flex-wrap items-start justify-between gap-2 border-b border-[var(--border-subtle)] pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">
              {payload.label || "Predicted DTL Test-Time Cost Saving — Selected Die"}
            </h3>
            <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)] px-2 py-0.5 text-[10px] font-mono uppercase tracking-wide text-[var(--text-muted)]">
              {payload.status}
            </span>
          </div>
          {scope && (
            <div className="mt-1.5 flex flex-wrap gap-1.5 text-xs font-mono" data-testid="cost-savings-selected-scope">
              <span className="rounded bg-[var(--bg-panel-secondary)] px-2 py-0.5 text-[var(--text-secondary)] border border-[var(--border-subtle)]">
                Cat: {scope.category || "NORMAL"}
              </span>
              <span className="rounded bg-[var(--bg-panel-secondary)] px-2 py-0.5 text-[var(--text-secondary)] border border-[var(--border-subtle)]">
                Lot: {scope.lot_id}
              </span>
              <span className="rounded bg-[var(--bg-panel-secondary)] px-2 py-0.5 text-[var(--text-primary)] border border-[var(--border-muted)] font-bold">
                Die: {scope.die_id}
              </span>
              {scope.production_month && scope.production_month !== "three-month" && (
                <span className="rounded bg-amber-500/10 px-2 py-0.5 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                  Month: {scope.production_month}
                </span>
              )}
            </div>
          )}
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 text-[11px] text-[var(--text-muted)]" data-testid="cost-savings-disclaimer">
            <span className="font-medium text-amber-600 dark:text-amber-400">Counterfactual estimate — not measured ATE savings.</span>
            <span>Predictions should be validated on the ATE before production deployment.</span>
          </div>
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5 text-sm">
        {/* 1. Estimated Money Saved */}
        <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
          <dt className="text-xs font-medium text-[var(--text-muted)] flex items-center justify-between">
            <span>Estimated Money Saved</span>
            <span className="text-[10px] text-[var(--text-muted)] cursor-help" title="Model-estimated savings per die on ATE">ⓘ</span>
          </dt>
          <dd className="mt-1 flex flex-col">
            <span className="font-mono text-xl font-bold text-[var(--accent)]" data-testid="cost-savings-total">
              {moneyFormatted.primary}
            </span>
            {moneyFormatted.centsSub && (
              <span className="font-mono text-[11px] text-[var(--text-muted)]">
                ({moneyFormatted.centsSub})
              </span>
            )}
          </dd>
        </div>

        {/* 2. Test Time Reduced */}
        <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
          <dt className="text-xs font-medium text-[var(--text-muted)] flex items-center justify-between">
            <span>Test Time Reduced</span>
            <span className="text-[10px] text-[var(--text-muted)] cursor-help" title="Percentage reduction in total test execution time">ⓘ</span>
          </dt>
          <dd className="mt-1 font-mono text-xl font-bold text-emerald-600 dark:text-emerald-400" data-testid="cost-savings-pct">
            {agg.predicted_time_saved_pct.toFixed(1)}%
          </dd>
        </div>

        {/* 3. Current Test Time */}
        <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
          <dt className="text-xs font-medium text-[var(--text-muted)]">Current Test Time</dt>
          <dd className="mt-1 font-mono text-lg font-semibold text-[var(--text-primary)]" data-testid="cost-savings-baseline">
            {seconds(agg.total_baseline_test_time_s)}
          </dd>
        </div>

        {/* 4. New Test Time */}
        <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
          <dt className="text-xs font-medium text-[var(--text-muted)]">New Test Time</dt>
          <dd className="mt-1 font-mono text-lg font-semibold text-[var(--text-primary)]" data-testid="cost-savings-dtl">
            {seconds(agg.total_dtl_test_time_s)}
          </dd>
        </div>

        {/* 5. Predicted Test Conditions to Skip */}
        <div className="rounded-md bg-[var(--bg-panel-secondary)] p-3 border border-[var(--border-subtle)]">
          <dt className="text-xs font-medium text-[var(--text-muted)] flex items-center justify-between">
            <span>Predicted Conditions to Skip</span>
            <span className="text-[10px] text-[var(--text-muted)] cursor-help" title="Number of conditions that can safely be bypassed">ⓘ</span>
          </dt>
          <dd className="mt-1 font-mono text-lg font-semibold text-[var(--text-primary)]" data-testid="cost-savings-skips">
            {agg.records_with_predicted_skip}
          </dd>
        </div>
      </dl>

      {/* Expandable Explanation Section */}
      <details className="mt-4 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)] p-3 text-xs text-[var(--text-secondary)]" data-testid="cost-savings-explanation">
        <summary className="cursor-pointer font-medium text-[var(--text-primary)] hover:text-[var(--accent)] transition-colors">
          What do these numbers mean?
        </summary>
        <div className="mt-3 space-y-3 border-t border-[var(--border-subtle)] pt-3">
          <dl className="space-y-2">
            <div>
              <dt className="font-semibold text-[var(--text-primary)]">Estimated Money Saved</dt>
              <dd className="text-[var(--text-secondary)] mt-0.5">
                Estimated tester cost saved for the selected die if the recommended test conditions are adopted. This is a model-based estimate, not measured ATE savings.
              </dd>
            </div>

            <div>
              <dt className="font-semibold text-emerald-600 dark:text-emerald-400">Test Time Reduced</dt>
              <dd className="text-[var(--text-secondary)] mt-0.5">
                Estimated percentage reduction in test time compared with the current test approach.
              </dd>
            </div>

            <div>
              <dt className="font-semibold text-[var(--text-primary)]">Current Test Time</dt>
              <dd className="text-[var(--text-secondary)] mt-0.5">
                Estimated time required to test the selected die using the current test conditions.
              </dd>
            </div>

            <div>
              <dt className="font-semibold text-[var(--text-primary)]">New Test Time</dt>
              <dd className="text-[var(--text-secondary)] mt-0.5">
                Estimated time required after applying the recommended test limits and predicted condition skips.
              </dd>
            </div>

            <div>
              <dt className="font-semibold text-[var(--text-primary)]">Predicted Test Conditions to Skip</dt>
              <dd className="text-[var(--text-secondary)] mt-0.5">
                Number of individual parametric test-condition evaluations that the model predicts can be omitted for this die.
              </dd>
            </div>
          </dl>

          <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-3 font-mono text-[11px] text-[var(--text-secondary)]">
            <p className="font-semibold text-[var(--accent)] uppercase tracking-wide text-[10px]">Example:</p>
            <ul className="mt-1 space-y-0.5">
              <li>Current test time: 1.40 s</li>
              <li>New test time: 0.50 s</li>
              <li>Estimated reduction: 64.3%</li>
              <li>Predicted conditions to skip: 6</li>
            </ul>
            <p className="mt-2 text-[var(--text-muted)] font-sans text-[11px] italic">
              In other words, the model predicts that this die could require less tester time by avoiding test-condition evaluations that are predicted to provide limited additional value.
            </p>
          </div>
        </div>
      </details>
    </section>
  );
}
