import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type {
  CandidateSetEntry,
  Decision,
  DieConditionsResponse,
  DieDistributionResponse,
  DieMeasurementResponse,
  DTLRecommendation,
} from "@/api/types";
import { ConditionTable } from "@/components/measurement/ConditionTable";
import { DecisionContextPanel } from "@/components/measurement/DecisionContextPanel";
import { MeasurementCard } from "@/components/measurement/MeasurementCard";
import { MeasurementDistribution } from "@/components/measurement/MeasurementDistribution";
import { ObservedRangeChart } from "@/components/measurement/ObservedRangeChart";
import { DecisionCard } from "@/components/recommendation/DecisionCard";

/** Explicit test fixtures — not production silicon values. */
const FIXTURE_MEASUREMENT: DieMeasurementResponse = {
  lot_id: "TEST_LOT",
  die_id: "TEST_DIE",
  parameter: "ir_drop",
  domain: "core",
  unit: "mV",
  observed_value: 24.45,
  observed_value_rule: "median_over_patterns",
  source_classification: "SYNTHETIC",
  dataset_version: "DTL_DATASET_V1",
  found: true,
  disclaimer: "Synthetic dataset measurement — not production silicon truth.",
};

const FIXTURE_DISTRIBUTION: DieDistributionResponse = {
  lot_id: "TEST_LOT",
  die_id: "TEST_DIE",
  parameter: "ir_drop",
  domain: "core",
  unit: "mV",
  scope: "die",
  n: 200,
  min: 21.98,
  median: 24.45,
  p95: 25.85,
  max: 31.69,
  source_classification: "SYNTHETIC",
  dataset_version: "DTL_DATASET_V1",
  stats_method: "phase3_compute_dist_stats",
  found: true,
};

const FIXTURE_PARAM_CONDITIONS: DieConditionsResponse = {
  lot_id: "TEST_LOT",
  die_id: "TEST_DIE",
  parameter: "VMIN",
  domain: "parametric",
  unit: "V",
  source_classification: "SYNTHETIC",
  dataset_version: "DTL_PARAMETRIC_DATASET_V1",
  found: true,
  conditions: [
    {
      condition_id: "COND_RT_NOM",
      temperature_c: 25,
      vdd_applied: 1.0,
      test_mode: "NOMINAL",
      measurement_value: 0.94,
      unit: "V",
      pass_fail_condition: "F",
    },
    {
      condition_id: "COND_HOT_NOM",
      temperature_c: 85,
      vdd_applied: 1.0,
      test_mode: "HOT",
      measurement_value: 1.02,
      unit: "V",
      pass_fail_condition: "F",
    },
  ],
};

function fixtureRec(decision: Decision): DTLRecommendation {
  return {
    request_id: "test-req",
    lot_id: "TEST_LOT",
    die_id: "TEST_DIE",
    parameter: "ir_drop",
    test_id: "T1",
    unit: "mV",
    direction: "upper",
    current_limit: 25,
    recommended_limit: decision === "KEEP_CURRENT" ? 25 : 30,
    decision,
    ml_score: 0.5,
    ml_rank: 1,
    n_candidates: 2,
    model_id: "core_gru",
    source_status: "SYNTHETIC",
    simulation_evidence: {
      evidence_origin: "SIMULATOR_DERIVED",
      population_level_aggregate: true,
      parameter: "ir_drop",
      candidate_limit: 30,
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
    explanation: {},
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
  };
}

const FIXTURE_CANDIDATES: CandidateSetEntry[] = [
  {
    parameter: "ir_drop",
    candidate_limit: 30,
    ml_rank: 1,
    unit: "mV",
  },
  {
    parameter: "ir_drop",
    candidate_limit: 25,
    ml_rank: 2,
    unit: "mV",
  },
];

describe("MeasurementCard", () => {
  it("renders observed value", () => {
    render(<MeasurementCard measurement={FIXTURE_MEASUREMENT} patternCount={200} />);
    expect(screen.getByTestId("observed-value")).toHaveTextContent("24.45 mV");
  });

  it("renders unit", () => {
    render(<MeasurementCard measurement={FIXTURE_MEASUREMENT} />);
    expect(screen.getByTestId("observed-value").textContent).toMatch(/mV/);
  });

  it("shows SYNTHETIC DATASET label", () => {
    render(<MeasurementCard measurement={FIXTURE_MEASUREMENT} />);
    expect(screen.getByTestId("synthetic-label")).toHaveTextContent("SYNTHETIC DATASET");
    expect(screen.queryByText(/Actual Chip Reading/i)).not.toBeInTheDocument();
  });

  it("renders empty state when measurement missing", () => {
    render(
      <MeasurementCard
        measurement={{ ...FIXTURE_MEASUREMENT, found: false, observed_value: null }}
      />,
    );
    expect(screen.getByText(/Measurement unavailable/i)).toBeInTheDocument();
    expect(
      screen.getByText(/No measurement is available for the selected lot, die, and parameter/i),
    ).toBeInTheDocument();
  });

  it("renders loading state", () => {
    render(<MeasurementCard measurement={null} loading />);
    expect(screen.getByText(/Loading measurement/i)).toBeInTheDocument();
  });

  it("renders sanitized API error", () => {
    render(
      <MeasurementCard
        measurement={null}
        error="Die not found for lot."
      />,
    );
    expect(screen.getByText(/Unable to load measurement data/i)).toBeInTheDocument();
    expect(screen.getByText(/Die not found for lot/i)).toBeInTheDocument();
  });
});

describe("MeasurementDistribution", () => {
  it("renders min/median/P95/max", () => {
    render(<MeasurementDistribution distribution={FIXTURE_DISTRIBUTION} />);
    expect(screen.getByTestId("dist-min")).toHaveTextContent("21.98 mV");
    expect(screen.getByTestId("dist-median")).toHaveTextContent("24.45 mV");
    expect(screen.getByTestId("dist-p95")).toHaveTextContent("25.85 mV");
    expect(screen.getByTestId("dist-max")).toHaveTextContent("31.69 mV");
  });

  it("renders empty state when distribution missing", () => {
    render(
      <MeasurementDistribution
        distribution={{ ...FIXTURE_DISTRIBUTION, found: false, min: null, median: null, p95: null, max: null }}
      />,
    );
    expect(screen.getByText(/Distribution unavailable/i)).toBeInTheDocument();
  });
});

describe("ObservedRangeChart markers", () => {
  it("keeps current limit and ML Top Candidate visually distinct", () => {
    render(
      <ObservedRangeChart
        distribution={FIXTURE_DISTRIBUTION}
        rec={fixtureRec("RECOMMEND")}
        candidates={FIXTURE_CANDIDATES}
      />,
    );
    expect(screen.getByTestId("legend-current-limit")).toBeInTheDocument();
    expect(screen.getByTestId("legend-ml-top-candidate")).toBeInTheDocument();
    expect(screen.getByTestId("legend-final-recommendation")).toBeInTheDocument();
    expect(screen.getByTestId("range-current-limit")).toBeInTheDocument();
  });
});

describe("DecisionContextPanel", () => {
  it("displays current and recommended DTL without ML-top as the decision", () => {
    render(
      <DecisionContextPanel rec={fixtureRec("RECOMMEND")} candidates={FIXTURE_CANDIDATES} />,
    );
    expect(screen.getByTestId("context-final-recommendation")).toHaveTextContent("30 mV");
    expect(screen.getByTestId("context-current-limit")).toHaveTextContent("25 mV");
    expect(screen.getByText("Recommended DTL")).toBeInTheDocument();
    expect(screen.getByTestId("context-action")).toHaveTextContent(/Change DTL/i);
    expect(screen.queryByText("AI Candidate")).not.toBeInTheDocument();
    expect(screen.queryByTestId("context-ml-top-candidate")).not.toBeInTheDocument();
  });

  it.each([
    "KEEP_CURRENT",
    "RECOMMEND",
    "REVIEW_REQUIRED",
    "REJECT",
  ] as const)("renders decision %s", (decision) => {
    render(<DecisionContextPanel rec={fixtureRec(decision)} candidates={FIXTURE_CANDIDATES} />);
    expect(screen.getByTestId("context-decision")).toHaveTextContent(decision);
    render(<DecisionCard decision={decision} />);
    expect(screen.getByLabelText(`Final decision ${decision}`)).toBeInTheDocument();
  });
});

describe("ConditionTable", () => {
  it("shows not-condition-aware for Core", () => {
    render(
      <ConditionTable
        conditions={{
          ...FIXTURE_PARAM_CONDITIONS,
          domain: "core",
          parameter: "ir_drop",
          found: false,
          reason: "not_condition_aware",
          conditions: [],
        }}
      />,
    );
    expect(screen.getByTestId("not-condition-aware")).toHaveTextContent(/Not condition-aware/i);
    expect(screen.getByText(/Core measurements are not condition-aware/i)).toBeInTheDocument();
  });

  it("renders parametric condition table", () => {
    render(<ConditionTable conditions={FIXTURE_PARAM_CONDITIONS} />);
    expect(screen.getByTestId("condition-table")).toBeInTheDocument();
    expect(screen.getByText("COND_RT_NOM")).toBeInTheDocument();
    expect(screen.getByText("COND_HOT_NOM")).toBeInTheDocument();
  });
});

describe("stale selection clearing", () => {
  it("does not keep previous measurement when props clear", () => {
    const { rerender } = render(
      <MeasurementCard measurement={FIXTURE_MEASUREMENT} patternCount={200} />,
    );
    expect(screen.getByTestId("observed-value")).toHaveTextContent("24.45 mV");
    rerender(<MeasurementCard measurement={null} loading />);
    expect(screen.queryByTestId("observed-value")).not.toBeInTheDocument();
    expect(screen.getByText(/Loading measurement/i)).toBeInTheDocument();
  });
});

describe("retry affordance", () => {
  it("exposes retry on measurement error", () => {
    const onRetry = vi.fn();
    render(<MeasurementCard measurement={null} error="Unable to load measurement data." onRetry={onRetry} />);
    screen.getByRole("button", { name: /retry/i }).click();
    expect(onRetry).toHaveBeenCalled();
  });
});
