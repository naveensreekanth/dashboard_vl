"use client";

import { useState } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DetailPopup } from "@/components/common/DetailPopup";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { LoadingState } from "@/components/common/LoadingState";
import { useMaintenance, useMaintenanceTester } from "@/hooks/useMaintenance";
import { formatNumber, formatTime } from "@/lib/utils";
import type { MaintenanceAsset, MaintenanceSeverity } from "@/types/maintenance";
import { SEVERITY_COLORS } from "@/types/maintenance";

export function PredictiveMaintenanceCard() {
  const { data, isLoading, isError, refetch } = useMaintenance();
  const [selectedTesterId, setSelectedTesterId] = useState<string | null>(null);
  const detailQuery = useMaintenanceTester(selectedTesterId);

  if (isLoading && !data) {
    return <LoadingState label="Loading maintenance predictions…" />;
  }

  if (isError && !data) {
    return (
      <ErrorState
        message="Unable to load predictive maintenance from the API."
        onRetry={() => void refetch()}
      />
    );
  }

  if (!data?.model_available) {
    return (
      <div className="vl-card flex flex-col gap-2.5 overflow-hidden p-4">
        <div className="text-[13px] font-semibold text-[#f2f7fc]">Predictive Maintenance</div>
        <div className="rounded-[6px] border border-[rgba(107,193,242,0.2)] bg-[rgba(107,193,242,0.06)] px-3 py-4 text-center text-[12px] text-[#b7c9dd]">
          Prediction unavailable
        </div>
      </div>
    );
  }

  const assets = data.assets ?? [];
  if (assets.length === 0) {
    return (
      <div className="vl-card flex flex-col gap-2.5 overflow-hidden p-4">
        <div className="text-[13px] font-semibold text-[#f2f7fc]">Predictive Maintenance</div>
        <EmptyState message="No maintenance predictions yet." />
      </div>
    );
  }

  const flagged = data.flagged_count;

  return (
    <>
      <div className="vl-card flex flex-col gap-2.5 overflow-hidden p-4">
        <div className="flex items-start justify-between">
          <div className="text-[13px] font-semibold text-[#f2f7fc]">Predictive Maintenance</div>
          <span className="rounded bg-[var(--amber-dim)] px-[7px] py-0.5 text-[10px] font-semibold text-[var(--amber)]">
            {flagged} flagged
          </span>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {(["healthy", "watch", "warning", "critical", "offline"] as MaintenanceSeverity[]).map(
            (sev) => (
              <span
                key={sev}
                className="rounded px-1.5 py-0.5 text-[9px] uppercase tracking-[0.06em]"
                style={{
                  color: SEVERITY_COLORS[sev],
                  background: `${SEVERITY_COLORS[sev]}22`,
                }}
              >
                {sev}
              </span>
            ),
          )}
        </div>

        <div>
          {assets.map((asset) => (
            <button
              key={asset.asset_id}
              type="button"
              onClick={() => asset.tester_id && setSelectedTesterId(asset.tester_id)}
              className="flex w-full items-center justify-between border-t border-[var(--line)] py-[7px] text-left text-[11.5px] first:border-t-0 transition-[background-color] duration-200 hover:bg-[var(--panel-2)]"
            >
              <div className="min-w-0 flex-1 pr-2">
                <div className="truncate text-[var(--text)]">{asset.name}</div>
                <div className="mt-0.5 text-[10px] uppercase tracking-[0.06em]" style={{ color: SEVERITY_COLORS[asset.severity] }}>
                  {asset.severity}
                </div>
                {asset.model_available && asset.health_pct != null ? (
                  <div className="mt-[3px] h-[5px] w-16 overflow-hidden rounded-[3px] bg-[var(--line)]">
                    <div
                      className="h-full rounded-[3px]"
                      style={{
                        width: `${Math.max(0, Math.min(100, asset.health_pct))}%`,
                        background: SEVERITY_COLORS[asset.severity],
                      }}
                    />
                  </div>
                ) : null}
              </div>
              <AssetValue asset={asset} />
            </button>
          ))}
        </div>

        <div className="text-[11.5px] leading-relaxed text-[var(--muted)]">
          RUL / failure probability from Python sklearn ensemble on tester telemetry features.
        </div>
      </div>

      {selectedTesterId ? (
        <MaintenanceDetailDrawer
          testerId={selectedTesterId}
          loading={detailQuery.isLoading}
          detail={detailQuery.data}
          onClose={() => setSelectedTesterId(null)}
        />
      ) : null}
    </>
  );
}

function AssetValue({ asset }: { asset: MaintenanceAsset }) {
  if (!asset.model_available || asset.severity === "unavailable" || asset.health_pct == null) {
    return (
      <div className="font-mono text-[11px] font-semibold text-[var(--muted)]">
        Prediction unavailable
      </div>
    );
  }
  return (
    <div className="text-right">
      <div
        className="font-mono font-semibold"
        style={{ color: SEVERITY_COLORS[asset.severity] }}
      >
        {Math.round(asset.health_pct)}%
      </div>
      <div className="font-mono text-[10px] text-[var(--muted-2)]">
        RUL {asset.rul_days != null ? `${formatNumber(asset.rul_days, 1)}d` : "—"}
      </div>
    </div>
  );
}

function MaintenanceDetailDrawer({
  testerId,
  loading,
  detail,
  onClose,
}: {
  testerId: string;
  loading: boolean;
  detail: ReturnType<typeof useMaintenanceTester>["data"];
  onClose: () => void;
}) {
  const series = (detail?.health_series ?? [])
    .filter((p) => p.health_score != null)
    .map((p) => ({
      t: formatTime(p.timestamp),
      v: p.health_score as number,
      fp: p.failure_probability,
    }));

  return (
    <DetailPopup
      eyebrow="Maintenance Detail"
      title={detail?.name ?? testerId}
      onClose={onClose}
      wide
    >
      {loading || !detail ? (
        <LoadingState label="Loading tester health…" />
      ) : !detail.model_available ? (
        <div className="rounded-[var(--radius-card)] border border-[var(--line)] bg-[var(--popup-panel)] px-3 py-6 text-center text-[12px] text-[var(--muted)]">
          Prediction unavailable
        </div>
      ) : (
        <>
          <div
            className="mb-3 text-[11px] uppercase tracking-[0.08em]"
            style={{ color: SEVERITY_COLORS[detail.overall_severity] }}
          >
            Overall · {detail.overall_severity}
          </div>

          {detail.components.map((c) => (
            <div key={c.asset_id} className="vl-popup-tile mb-3 p-3 text-[11.5px]">
              <div className="mb-1 font-semibold">{c.component ?? c.name}</div>
              {c.model_available && c.health_pct != null ? (
                <dl className="grid grid-cols-[120px_1fr] gap-y-1">
                  <dt className="text-[var(--muted-2)]">Health</dt>
                  <dd className="font-mono">{formatNumber(c.health_pct, 1)}%</dd>
                  <dt className="text-[var(--muted-2)]">Failure Prob</dt>
                  <dd className="font-mono">
                    {c.failure_probability != null
                      ? `${formatNumber(c.failure_probability * 100, 1)}%`
                      : "—"}
                  </dd>
                  <dt className="text-[var(--muted-2)]">RUL</dt>
                  <dd className="font-mono">
                    {c.rul_days != null ? `${formatNumber(c.rul_days, 1)} days` : "—"}
                  </dd>
                  <dt className="text-[var(--muted-2)]">Confidence</dt>
                  <dd className="font-mono">
                    {c.confidence != null ? `${formatNumber(c.confidence * 100, 1)}%` : "—"}
                  </dd>
                  <dt className="text-[var(--muted-2)]">Severity</dt>
                  <dd className="font-mono uppercase">{c.severity}</dd>
                  <dt className="text-[var(--muted-2)]">Recommendation</dt>
                  <dd>{c.recommended_action ?? "—"}</dd>
                </dl>
              ) : (
                <div className="text-[var(--muted)]">Prediction unavailable</div>
              )}
            </div>
          ))}

          <div className="vl-label mb-2">Historical Health</div>
          <div className="vl-popup-tile mb-5 h-[180px] p-2">
            {series.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={series}>
                  <XAxis dataKey="t" tick={{ fill: "#56637A", fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#56637A", fontSize: 10 }} width={32} />
                  <Tooltip
                    contentStyle={{
                      background: "#0D131C",
                      border: "1px solid #2A3648",
                      fontSize: 11,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="v"
                    stroke="#F2B155"
                    fill="#F2B15533"
                    strokeWidth={1.6}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState message="No health history yet." />
            )}
          </div>

          <div className="vl-label mb-2">Maintenance History</div>
          <div className="flex flex-col gap-2">
            {(detail.history ?? []).map((h) => (
              <div key={h.history_id} className="vl-popup-tile px-3 py-2 text-[11px]">
                <div className="mb-1 flex justify-between text-[var(--muted-2)]">
                  <span className="uppercase">{h.severity ?? h.event_type}</span>
                  <span className="font-mono">{formatTime(h.created_at)}</span>
                </div>
                <div className="text-[var(--text)]">{h.detail}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </DetailPopup>
  );
}
