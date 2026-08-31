import Image from "next/image";

type VerilumenBrandProps = {
  /** Compact header mark vs slightly larger login treatment */
  size?: "header" | "auth";
};

/**
 * Compact brand lockup: V emblem + VERILUMEN + ATE INTELLIGENCE.
 * Full banner / tagline omitted for header density.
 */
export function VerilumenBrand({ size = "header" }: VerilumenBrandProps) {
  const markH = size === "auth" ? 58 : 50;

  return (
    <div className="flex items-center gap-3.5">
      <div
        className="relative flex shrink-0 items-center justify-center rounded-[10px] border border-[var(--line)]"
        style={{
          width: markH + 10,
          height: markH + 10,
          background:
            "radial-gradient(circle at 50% 35%, rgba(107,193,242,0.16), transparent 65%), #070c14",
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.06)",
        }}
      >
        <Image
          src="/branding/verilumen-mark.png"
          alt="VERILUMEN"
          width={64}
          height={64}
          className="object-contain"
          style={{ height: markH, width: markH }}
          priority
        />
      </div>
      <div className="min-w-0 leading-tight">
        <div
          className={
            size === "auth"
              ? "font-display text-[17px] font-bold uppercase tracking-[0.18em] text-[#f2f6fb]"
              : "font-display text-[14px] font-bold uppercase tracking-[0.18em] text-[#f2f6fb]"
          }
        >
          Verilumen
        </div>
        <div
          className={
            size === "auth"
              ? "mt-1.5 font-display text-[21px] font-semibold uppercase tracking-[0.16em] text-[var(--cyan)]"
              : "mt-1 font-display text-[18px] font-semibold uppercase tracking-[0.16em] text-[var(--cyan)]"
          }
        >
          ATE Intelligence
        </div>
      </div>
    </div>
  );
}
