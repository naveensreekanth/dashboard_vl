"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DetailPopup } from "@/components/common/DetailPopup";
import { LoadingState } from "@/components/common/LoadingState";
import { useKpiDetail } from "@/hooks/useKpis";
import { useKpiHistory } from "@/hooks/useKpiHistory";
import { formatNumber, formatTime } from "@/lib/utils";

/**
 * KPI drill-down as a brighter centered popup.
 */
export function KpiDetailDrawer({
  kpiId,
  onClose,
}: {
  kpiId: string;
  onClose: () => void;
}) {
  const detailQuery = useKpiDetail(kpiId);
  const { history, isLoading: histLoading } = useKpiHistory(kpiId, 48);
  const kpi = detailQuery.data;
  const accent = kpi?.accent ?? "#7DD3FC";
  const digits = kpi && kpi.unit === "%" && kpi.value > 90 ? 2 : 1;
  const chartData = history.map((p) => ({
    t: p.timestamp,
    v: p.value,
    label: formatTime(p.timestamp),
  }));

  return (
    <DetailPopup
      eyebrow="KPI Analytics"
      title={kpi?.name ?? "Loading…"}
      onClose={onClose}
      wide
    >
      {detailQuery.isLoading || !kpi ? (
        <LoadingState label="Loading KPI detail…" />
      ) : detailQuery.isError ? (
        <div className="rounded-[8px] border border-[var(--red)]/50 bg-[var(--red-dim)] px-3 py-4 text-[12px] text-[var(--red)]">
          Unable to load KPI detail. Close and try again.
        </div>
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-2.5 sm:grid-cols-3">
            <Tile
              label="Current Value"
              value={`${formatNumber(kpi.value, digits)}${kpi.unit}`}
              accent={accent}
            />
            <Tile
              label="Previous"
              value={`${formatNumber(kpi.previous_value, digits)}${kpi.unit}`}
            />
            <Tile
              label="Baseline"
              value={`${formatNumber(kpi.baseline, digits)}${kpi.unit}`}
            />
            <Tile
              label="Target"
              value={`${formatNumber(kpi.target, digits)}${kpi.unit}`}
              accent="var(--cyan)"
            />
            <Tile
              label="Improvement"
              value={`${kpi.improvement >= 0 ? "+" : ""}${formatNumber(kpi.improvement, digits)}${kpi.unit}`}
              accent={kpi.improvement >= 0 ? "var(--green)" : "var(--red)"}
            />
            <Tile
              label="Trend / Status"
              value={`${kpi.trend} · ${String(kpi.status).replace(/_/g, " ")}`}
              accent="var(--amber)"
            />
          </div>

          <div className="mb-4 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
            <Tile label="Lots" value={String(kpi.lots)} compact />
            <Tile label="Wafers" value={String(kpi.wafers)} compact />
            <Tile label="Testers" value={String(kpi.testers)} compact />
            <Tile label="Sites" value={String(kpi.sites)} compact />
          </div>

          <div className="vl-popup-label mb-2">Historical Trend</div>
          <div className="vl-popup-tile mb-5 h-[220px] p-2.5">
            {histLoading ? (
              <LoadingState label="Loading history…" />
            ) : chartData.length === 0 ? (
              <div className="flex h-full items-center justify-center text-[12px] text-[#c5d8ec]">
                No history points yet
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <CartesianGrid stroke="rgba(107,193,242,0.18)" strokeDasharray="3 3" />
                  <XAxis dataKey="label" tick={{ fill: "#A8C6E4", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#A8C6E4", fontSize: 10 }} width={36} />
                  <Tooltip
                    contentStyle={{
                      background: "#152033",
                      border: "1px solid rgba(107,193,242,0.4)",
                      fontSize: 11,
                      color: "#F5F9FF",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="v"
                    stroke={accent}
                    fill={`${accent}55`}
                    strokeWidth={2.2}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="vl-popup-label mb-2">Recent Events</div>
          <div className="flex flex-col gap-2">
            {(kpi.recent_events ?? []).length === 0 ? (
              <div className="vl-popup-tile px-3 py-3 text-[11px] text-[#c5d8ec]">
                No recent events for this KPI
              </div>
            ) : (
              (kpi.recent_events ?? []).map((ev) => (
                <div key={ev.event_id} className="vl-popup-tile px-3 py-2.5 text-[11px]">
                  <div className="mb-1 flex justify-between text-[#b7d4f0]">
                    <span
                      className={`font-semibold uppercase ${
                        ev.tag === "pass"
                          ? "text-[var(--green)]"
                          : ev.tag === "warn"
                            ? "text-[var(--amber)]"
                            : "text-[var(--cyan)]"
                      }`}
                    >
                      {ev.tag}
                    </span>
                    <span className="font-mono text-[#d7e8f8]">{formatTime(ev.timestamp)}</span>
                  </div>
                  <div className="text-[#f2f7fc]">{ev.text}</div>
                </div>
              ))
            )}
          </div>

          <p className="mt-4 text-[12px] leading-relaxed text-[#d5e6f7]">{kpi.description}</p>
          <p className="mt-2 font-mono text-[10px] text-[#9ec9ef]">
            Last updated {formatTime(kpi.timestamp)}
          </p>
        </>
      )}
    </DetailPopup>
  );
}

function Tile({
  label,
  value,
  accent,
  compact,
}: {
  label: string;
  value: string;
  accent?: string;
  compact?: boolean;
}) {
  return (
    <div className={`vl-popup-tile ${compact ? "px-2.5 py-2" : "px-3 py-2.5"}`}>
      <div className="vl-popup-tile-label">{label}</div>
      <div
        className={`vl-popup-tile-value font-mono font-semibold ${compact ? "text-[13px]" : "text-[16px]"} mt-1`}
        style={{ color: accent }}
      >
        {value}
      </div>
    </div>
  );
}
