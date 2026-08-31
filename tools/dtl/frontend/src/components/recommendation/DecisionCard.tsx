import type { Decision } from "@/api/types";
import { formatDecision } from "@/utils/formatDecision";

function DecisionIcon({ icon }: { icon: "check" | "minus" | "exclamation" | "x" }) {
  const map = {
    check: "✓",
    minus: "−",
    exclamation: "!",
    x: "✗",
  };
  return (
    <span className="text-2xl font-bold" aria-hidden="true">
      {map[icon]}
    </span>
  );
}

export function DecisionCard({ decision }: { decision: Decision }) {
  const display = formatDecision(decision);
  return (
    <section
      className={`rounded-lg border-2 bg-gray-900 p-4 ${display.borderClass}`}
      aria-label={`Final decision ${display.label}`}
    >
      <div className="flex items-center gap-3">
        <div className={display.textClass}>
          <DecisionIcon icon={display.icon} />
        </div>
        <div>
          <p className={`text-lg font-semibold ${display.textClass}`}>{display.label}</p>
          <p className="text-sm text-gray-400">{display.description}</p>
        </div>
      </div>
    </section>
  );
}
