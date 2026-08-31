"use client";

import type { KpiCard } from "@/types/api";
import { formatNumber } from "@/lib/utils";

export function OptimizationMetricCard({ card }: { card: KpiCard }) {
  const digits = card.unit === "%" && card.value > 90 ? 2 : 1;

  return (
    <div className="relative flex flex-col gap-2.5 rounded border border-[var(--line)] bg-[var(--panel)] p-[17px]">
      <div className="flex items-start justify-between">
        <div className="text-[12.5px] font-semibold tracking-[0.01em]">{card.title}</div>
        <span
          className={`rounded-full px-[7px] py-0.5 text-[10px] font-semibold tracking-[0.02em] ${
            card.trend === "up"
              ? "bg-[var(--green-dim)] text-[var(--green)]"
              : "bg-[var(--red-dim)] text-[var(--red)]"
          }`}
        >
          {card.trend === "up" ? "▲" : "▼"} trend
        </span>
      </div>
      <div className="font-display text-[28px] font-bold">
        {formatNumber(card.value, digits)}
        <span className="ml-0.5 text-[14px] font-medium text-[var(--muted)]">{card.unit}</span>
      </div>
      <div className="text-[11.5px] leading-relaxed text-[var(--muted)]">{card.description}</div>
    </div>
  );
}
