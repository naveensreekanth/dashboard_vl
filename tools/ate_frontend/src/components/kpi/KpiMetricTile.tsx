"use client";

import type { ReactNode } from "react";
import { AnimatedNumber } from "@/components/common/AnimatedNumber";
import { formatNumber } from "@/lib/utils";

export type KpiMetricTileVariant = "cyan" | "green" | "red" | "violet";

export type KpiMetricTileMetric = {
  label: string;
  value: number | string;
  unit?: string;
  color?: string;
};

export type KpiMetricTileProps = {
  eyebrow?: string;
  title?: string;
  metrics: KpiMetricTileMetric[];
  variant?: KpiMetricTileVariant;
  size?: "card" | "popup";
  icon?: ReactNode;
};

/**
 * Flat in-card metric block — no nested grid / mini-card chrome.
 */
export function KpiMetricTile({
  eyebrow,
  title,
  metrics,
  size = "card",
  icon,
}: KpiMetricTileProps) {
  const large = size === "popup";

  return (
    <div className={large ? "py-1" : "py-0"}>
      {icon || eyebrow ? (
        <div className="mb-0.5 flex items-center gap-1">
          {icon ? <span className="shrink-0">{icon}</span> : null}
          {eyebrow ? (
            <div className={`vl-metric-eyebrow ${large ? "text-[9px]" : "text-[7px]"}`}>
              {eyebrow}
            </div>
          ) : null}
        </div>
      ) : null}
      {title ? (
        <div
          className={`vl-metric-title ${large ? "mb-1.5 text-[12px]" : "mb-0.5 text-[10px]"}`}
        >
          {title}
        </div>
      ) : null}
      <div className={`grid gap-1 ${metrics.length > 1 ? "grid-cols-2" : "grid-cols-1"}`}>
        {metrics.map((m) => (
          <MetricCell key={m.label} metric={m} large={large} />
        ))}
      </div>
    </div>
  );
}

function MetricCell({
  metric,
  large,
}: {
  metric: KpiMetricTileMetric;
  large: boolean;
}) {
  const numericValue =
    typeof metric.value === "number" ? metric.value : Number.NaN;
  const digits =
    metric.unit === "%" && Number.isFinite(numericValue) && numericValue > 90 ? 2 : 1;
  const isNumeric = typeof metric.value === "number" && Number.isFinite(metric.value);

  return (
    <div>
      <div className={`vl-metric-label ${large ? "text-[9px]" : "text-[7px]"}`}>
        {metric.label}
      </div>
      <div
        className={`vl-metric-value ${large ? "text-[18px]" : "text-[13px]"}`}
        style={{ color: metric.color ?? "var(--text-bright)" }}
      >
        {isNumeric ? (
          <>
            <AnimatedNumber value={numericValue} digits={digits} durationMs={650} />
            {metric.unit ? (
              <span
                className={`ml-0.5 font-medium text-[var(--text-dim)] ${
                  large ? "text-[11px]" : "text-[10px]"
                }`}
              >
                {metric.unit}
              </span>
            ) : null}
          </>
        ) : (
          <span className="text-[var(--text-bright)]">
            {typeof metric.value === "string"
              ? metric.value
              : formatNumber(Number(metric.value), digits)}
          </span>
        )}
      </div>
    </div>
  );
}
