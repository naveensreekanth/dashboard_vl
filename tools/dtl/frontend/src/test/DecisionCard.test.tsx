import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DecisionCard } from "@/components/recommendation/DecisionCard";

describe("DecisionCard", () => {
  it.each([
    "RECOMMEND",
    "KEEP_CURRENT",
    "REVIEW_REQUIRED",
    "REJECT",
  ] as const)("renders decision %s", (decision) => {
    render(<DecisionCard decision={decision} />);
    expect(screen.getByLabelText(`Final decision ${decision}`)).toBeInTheDocument();
    expect(screen.getByText(decision)).toBeInTheDocument();
  });
});
