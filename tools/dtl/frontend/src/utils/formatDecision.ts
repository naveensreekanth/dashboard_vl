import type { Decision } from "@/api/types";

export interface DecisionDisplay {
  label: string;
  description: string;
  borderClass: string;
  textClass: string;
  icon: "check" | "minus" | "exclamation" | "x";
}

const DECISION_MAP: Record<Decision, DecisionDisplay> = {
  RECOMMEND: {
    label: "RECOMMEND",
    description: "Change DTL limit",
    borderClass: "border-green-600",
    textClass: "text-green-400",
    icon: "check",
  },
  KEEP_CURRENT: {
    label: "KEEP_CURRENT",
    description: "Keep current DTL",
    borderClass: "border-blue-600",
    textClass: "text-blue-400",
    icon: "minus",
  },
  REVIEW_REQUIRED: {
    label: "REVIEW_REQUIRED",
    description: "Engineer review required",
    borderClass: "border-amber-600",
    textClass: "text-amber-400",
    icon: "exclamation",
  },
  REJECT: {
    label: "REJECT",
    description: "Request rejected",
    borderClass: "border-red-600",
    textClass: "text-red-400",
    icon: "x",
  },
};

export function formatDecision(decision: Decision): DecisionDisplay {
  return DECISION_MAP[decision];
}
