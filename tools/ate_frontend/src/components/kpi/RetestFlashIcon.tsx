import { Zap } from "lucide-react";

/** Purple circle + yellow flash mark matching the Retest AI Agent branding. */
export function RetestFlashIcon({ size = 12 }: { size?: number }) {
  const box = size + 8;

  return (
    <span
      className="inline-flex shrink-0 items-center justify-center rounded-full"
      style={{
        width: box,
        height: box,
        background: "linear-gradient(145deg, #8b5cf6, #6d28d9)",
        boxShadow: "0 0 0 1px rgba(167, 139, 250, 0.35)",
      }}
      aria-hidden
    >
      <Zap size={size} fill="#facc15" stroke="#facc15" strokeWidth={1.5} />
    </span>
  );
}
