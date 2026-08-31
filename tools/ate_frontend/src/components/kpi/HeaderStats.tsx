"use client";

import { AnimatedNumber } from "@/components/common/AnimatedNumber";
import type { HeaderStats as HeaderStatsType } from "@/types/api";

export function HeaderStats({ data }: { data: HeaderStatsType | null }) {
  return (
    <div className="flex flex-wrap gap-2">
      <Stat
        label="Lots In Test"
        numeric={data?.lots_in_test}
        digits={0}
        fallback="—"
      />
      <Stat
        label="Test Time Saved (24h)"
        numeric={data != null ? Math.round(data.test_time_saved_hours) : undefined}
        digits={0}
        suffix=" hrs"
        color="var(--green)"
        fallback="—"
      />
      <Stat
        label="Overall Yield"
        numeric={data?.overall_yield_pct}
        digits={1}
        suffix="%"
        color="var(--cyan)"
        fallback="—"
      />
    </div>
  );
}

function Stat({
  label,
  numeric,
  digits = 1,
  suffix,
  color,
  fallback,
}: {
  label: string;
  numeric?: number;
  digits?: number;
  suffix?: string;
  color?: string;
  fallback: string;
}) {
  const ready = numeric != null && Number.isFinite(numeric);
  return (
    <div className="vl-stat text-right">
      <div className="vl-label">{label}</div>
      <div
        className="font-mono mt-1 text-[22px] font-semibold tracking-tight"
        style={{ color: color ?? "var(--text)" }}
      >
        {ready ? (
          <>
            <AnimatedNumber value={numeric} digits={digits} durationMs={700} />
            {suffix ?? null}
          </>
        ) : (
          fallback
        )}
      </div>
    </div>
  );
}
