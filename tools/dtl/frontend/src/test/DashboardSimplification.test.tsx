import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { DTLRecommendation } from "@/api/types";
import { AdvancedEvidence } from "@/components/common/AdvancedEvidence";
import { ExplanationPanel } from "@/components/recommendation/ExplanationPanel";

const FIXTURE_REC = {
  request_id: "test-req",
  lot_id: "TEST_LOT",
  die_id: "TEST_DIE",
  parameter: "ir_drop",
  test_id: "T1",
  unit: "mV",
  direction: "upper",
  current_limit: 25,
  recommended_limit: 25,
  decision: "KEEP_CURRENT",
  ml_score: 0.5,
  ml_rank: 1,
  n_candidates: 2,
  model_id: "core_gru",
  source_status: "SYNTHETIC",
  simulation_evidence: {
    evidence_origin: "SIMULATOR_DERIVED",
    population_level_aggregate: true,
    parameter: "ir_drop",
    candidate_limit: 25,
    simulated_yield: 0.9,
    violation_rate: 0.1,
    borderline_rate: 0.05,
    objective_score: 0.8,
    worst_condition_yield: null,
    worst_condition_violation_rate: null,
    evaluated_conditions: null,
    found: true,
  },
  safety_result: { status: "PASS", checks: [] },
  evidence_level: "MODERATE_EVIDENCE",
  explanation: {
    text: "Keep current limit because ML top candidate did not clear safety and policy.",
    policy_reason: "KEEP_CURRENT selected by policy",
  },
  model_version: "test",
  checkpoint_id: null,
  dataset_version: "DTL_DATASET_V1",
  feature_registry_hash: null,
  simulation_config_version: null,
  policy_config_version: "v1",
  timestamp: "2026-01-01T00:00:00Z",
  core_available: true,
  parametric_available: false,
  cross_domain_available: false,
  evidence_origin: "SIMULATOR_DERIVED",
} as DTLRecommendation;

describe("ExplanationPanel primary", () => {
  it("shows backend explanation without inventing text", () => {
    render(<ExplanationPanel rec={FIXTURE_REC} primary />);
    expect(screen.getByText("Why this decision?")).toBeInTheDocument();
    expect(screen.getByTestId("why-decision-text")).toHaveTextContent(
      /Keep current limit because ML top candidate/,
    );
  });

  it("shows eligibility only from recorded safety checks", () => {
    const rec = {
      ...FIXTURE_REC,
      decision: "RECOMMEND" as const,
      recommended_limit: 30,
      safety_result: {
        status: "PASS",
        checks: [
          { name: "catalog_membership", passed: true, layer: 1, message: "in catalog", severity: "hard" },
          { name: "simulation_evidence", passed: true, layer: 2, message: "found", severity: "soft" },
        ],
      },
      explanation: {
        ...FIXTURE_REC.explanation,
        selected_simulated_yield: 0.9907,
        ml_rank: 1,
        yield_tie: false,
        selection_text: "Highest simulated yield among eligible candidates.",
        simulator_note: "Recommended based on simulator-derived evidence.",
      },
    };
    render(<ExplanationPanel rec={rec} primary />);
    expect(screen.getByText("Why was this DTL selected?")).toBeInTheDocument();
    expect(screen.getByTestId("why-eligibility")).toHaveTextContent("Test data is valid");
    expect(screen.getByTestId("why-eligibility")).toHaveTextContent("Simulation supports the selection");
    expect(screen.getByTestId("why-eligibility")).not.toHaveTextContent("Required conditions covered");
    expect(screen.getByTestId("why-simulated-yield")).toHaveTextContent("99.07%");
    expect(screen.queryByText("ML selected the optimal DTL")).not.toBeInTheDocument();
  });
});

describe("AdvancedEvidence", () => {
  it("hides technical content until expanded", () => {
    render(
      <AdvancedEvidence>
        <p>Safety Gate Hidden Content</p>
      </AdvancedEvidence>,
    );
    expect(screen.getByTestId("advanced-evidence")).toBeInTheDocument();
    expect(screen.queryByText("Safety Gate Hidden Content")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Advanced Evidence/i }));
    expect(screen.getByText("Safety Gate Hidden Content")).toBeInTheDocument();
  });
});
