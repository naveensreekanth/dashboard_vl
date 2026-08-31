import type { WaferDetail } from "@/types/api";
import { formatNumber } from "@/lib/utils";

export function YieldSummary({ wafer }: { wafer: WaferDetail | null }) {
  const bins = wafer?.bin_counts;
  return (
    <div className="flex flex-col justify-center gap-4">
      <div className="flex flex-wrap gap-3">
        <span className="vl-chip">
          Lot{" "}
          <b className="font-mono text-[13px] text-white">{wafer?.lot_id ?? "—"}</b>
        </span>
        <span className="vl-chip">
          Wafer{" "}
          <b className="font-mono text-[13px] text-white">{wafer?.wafer_id ?? "—"}</b>
        </span>
      </div>

      <div className="font-display text-[58px] font-bold leading-none tracking-[-0.02em] text-white">
        <span>{wafer ? formatNumber(wafer.yield_pct) : "—"}</span>
        <span className="ml-1.5 text-[22px] font-medium text-[var(--cyan)]">% yield</span>
      </div>

      <p className="max-w-[560px] text-[13.5px] leading-relaxed text-[#b7c9dd]">
        Per-die bin classification, re-scored continuously against test-correlation history.
        Marginal fails are re-evaluated before being counted against yield; confirmed escapes
        trigger an immediate limit review on the affected site.
      </p>

      <div className="mt-1 flex flex-wrap gap-2.5">
        <Bin swatch="var(--green)" label="Pass" value={bins?.pass} />
        <Bin swatch="var(--amber)" label="Retest" value={bins?.retest} />
        <Bin swatch="var(--red)" label="Fail" value={bins?.fail} />
        <Bin swatch="var(--cyan)" label="Reclassified" value={bins?.reclass} />
      </div>
    </div>
  );
}

function Bin({
  swatch,
  label,
  value,
}: {
  swatch: string;
  label: string;
  value?: number;
}) {
  return (
    <span className="vl-chip">
      <i
        className="inline-block rounded-[2px]"
        style={{ background: swatch, width: 10, height: 10 }}
      />
      {label}
      <b className="font-mono text-white">{value ?? "—"}</b>
    </span>
  );
}
