import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ThreeMonthAnalysisBundle } from "@/api/analysisTypes";
import { ThreeMonthDashboardPage } from "@/pages/ThreeMonthDashboardPage";
import * as endpoints from "@/api/endpoints";
import { findRecommendation, whySelectedText } from "@/utils/analysisDisplay";

const FIXTURE: ThreeMonthAnalysisBundle = {
  source: "artifacts/temporal/shared/phase_12_9_analysis",
  disclaimer:
    "This dashboard uses synthetic three-month production-like data for engineering validation and demonstration. Simulated yield is not a guarantee of production yield.",
  allowed_months: ["2026-01", "2026-02", "2026-03"],
  scorable_parameters: [
    "IR_DROP_MV",
    "THERMAL_C",
    "VMIN",
    "VMAX",
    "IDDQ",
    "SUPPLY_CURRENT",
    "CONTACT_RESISTANCE",
    "INTERCONNECT_RESISTANCE",
    "ON_RESISTANCE",
  ],
  non_scorable_parameters: ["SETUP_SLACK_PS", "HOLD_SLACK_PS", "TEST_TIME_MS"],
  non_scorable_note:
    "Recommendation is currently unavailable for SETUP/HOLD/TEST_TIME because these parameters do not have a candidate/objective path.",
  executive_summary: {
    parameters_recommendation_changed: ["IR_DROP_MV"],
    parameters_recommendation_stable: ["THERMAL_C", "VMIN"],
    what_ml_does: "The ranking model scores candidate DTLs and produces ML rankings.",
    yield_first_proof_example: {
      kind: "yield_first",
      production_month: "2026-02",
      parameter: "VMAX",
      parameter_display: "VMAX",
      lot_id: "DTL_EDGE_001",
      die_id: "DTL_EDGE_001_D001",
      winner: { candidate_limit: 1.05, simulated_yield: 0.508, ml_rank: 2, ml_score: 0.1 },
      loser_higher_ml: {
        candidate_limit: 1.08,
        simulated_yield: 0.311,
        ml_rank: 1,
        ml_score: 0.13,
      },
      statement: "Yield first",
    },
    ml_tie_break_proof_example: {
      kind: "ml_tie_break",
      production_month: "2026-01",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      tied_yield: 1,
      winner_limit: 50,
      tied_candidates: [
        {
          candidate_limit: 50,
          simulated_yield: 1,
          ml_score: 0.84,
          ml_rank: 1,
          is_selected: true,
        },
        {
          candidate_limit: 55,
          simulated_yield: 1,
          ml_score: 0.83,
          ml_rank: 2,
          is_selected: false,
        },
      ],
    },
  },
  primary_recommendations: [
    {
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      production_month: "2026-01",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      unit: "mV",
      current_limit: 25,
      recommended_limit: 50,
      recommendation_delta: 25,
      recommendation_delta_percent: 100,
      max_eligible_simulated_yield: 1,
      ml_score: 0.84,
      ml_rank: 1,
      model_used: "core_gru_temporal_v1",
      decision: "RECOMMEND",
      policy_reason: "max_simulated_yield_selected",
      yield_tie: true,
      selection_text: "Candidates had equivalent simulated yield",
      why_selected: "ML rank tie-break selected 50 mV",
      safety_status: "PASS",
      evidence_origin: "SIMULATOR_DERIVED_TEMPORAL_2026-01",
      sequence_id: "2026-01::DTL_NORM_001::DTL_NORM_001_D001",
      is_primary_die: true,
    },
    {
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      production_month: "2026-02",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      unit: "mV",
      current_limit: 25,
      recommended_limit: 72,
      max_eligible_simulated_yield: 1,
      ml_rank: 1,
      model_used: "core_gru_temporal_v1",
      decision: "RECOMMEND",
      yield_tie: true,
      safety_status: "PASS",
      is_primary_die: true,
    },
    {
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      production_month: "2026-03",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      unit: "mV",
      current_limit: 25,
      recommended_limit: 55,
      max_eligible_simulated_yield: 1,
      ml_rank: 1,
      model_used: "core_gru_temporal_v1",
      decision: "RECOMMEND",
      yield_tie: true,
      safety_status: "PASS",
      is_primary_die: true,
    },
    {
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      production_month: "2026-01",
      parameter: "thermal",
      parameter_display: "THERMAL_C",
      unit: "°C",
      current_limit: 60,
      recommended_limit: 92,
      max_eligible_simulated_yield: 1,
      ml_rank: 1,
      model_used: "core_gru_temporal_v1",
      decision: "KEEP_CURRENT",
      safety_status: "PASS",
      why_selected: "Current remains selected",
      is_primary_die: true,
    },
  ],
  all_recommendations: [],
  candidate_explanations: [
    {
      production_month: "2026-01",
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      candidate_limit: 50,
      simulated_yield: 1,
      safety_status: "PASS",
      eligible: true,
      in_policy_gate_set: true,
      ml_score: 0.84,
      ml_rank: 1,
      is_selected: true,
      is_current: false,
    },
    {
      production_month: "2026-01",
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      candidate_limit: 55,
      simulated_yield: 1,
      safety_status: "PASS",
      eligible: true,
      in_policy_gate_set: true,
      ml_score: 0.83,
      ml_rank: 2,
      is_selected: false,
    },
  ],
  temporal_changes: [
    {
      parameter_display: "IR_DROP_MV",
      recommendation_changed: true,
      jan_recommendation: 50,
      feb_recommendation: 72,
      mar_recommendation: 55,
    },
  ],
  same_die_analysis: [
    {
      lot_category: "NORMAL",
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      production_month: "2026-01",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      recommended_limit: 50,
      max_eligible_simulated_yield: 1,
      ml_rank: 1,
      observed_mean: 23.1,
    },
    {
      lot_category: "NORMAL",
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      production_month: "2026-02",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      recommended_limit: 72,
      max_eligible_simulated_yield: 1,
      ml_rank: 1,
      observed_mean: 28.2,
    },
    {
      lot_category: "NORMAL",
      lot_id: "DTL_NORM_001",
      die_id: "DTL_NORM_001_D001",
      production_month: "2026-03",
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      recommended_limit: 55,
      max_eligible_simulated_yield: 1,
      ml_rank: 1,
      observed_mean: 25.3,
    },
  ],
  model_traceability: [
    {
      parameter: "ir_drop",
      parameter_display: "IR_DROP_MV",
      model_expected: "core_gru_temporal_v1",
      models_observed: "core_gru_temporal_v1",
      routing_ok: true,
    },
  ],
  policy_proofs: { ml_tie_break_proofs: [], yield_first_proofs: [] },
  viz_recommended_dtl_by_month: [],
  doc_reference: "docs/PHASE_12_9_THREE_MONTH_RECOMMENDATION_ANALYSIS.md",
  artifact_reference: "artifacts/temporal/shared/phase_12_9_analysis/",
  used_uploaded_measurements: true,
  used_static_three_month_measurements: false,
  data_provenance: "Analysis generated from uploaded test data",
  analysis_session_id: "sess-test-1",
  die_level_identities: {
    months: ["2026-01", "2026-02", "2026-03"],
    categories: ["NORMAL", "SCRATCH", "EDGE", "CENTER"],
    lots_by_category: {
      NORMAL: ["DTL_NORM_001", "DTL_NORM_002"],
      SCRATCH: ["DTL_SCRATCH_001"],
      EDGE: ["DTL_EDGE_003", "DTL_EDGE_001"],
      CENTER: ["DTL_CENTER_001"],
    },
    dies_by_lot: {
      DTL_NORM_001: ["DTL_NORM_001_D001", "DTL_NORM_001_D002"],
      DTL_NORM_002: ["DTL_NORM_002_D001"],
      DTL_SCRATCH_001: ["DTL_SCRATCH_001_D001"],
      DTL_EDGE_003: ["DTL_EDGE_003_D001", "DTL_EDGE_003_D025"],
      DTL_EDGE_001: ["DTL_EDGE_001_D001"],
      DTL_CENTER_001: ["DTL_CENTER_001_D001"],
    },
    counts: {
      lots: 20,
      dies: 1000,
      by_category_dies: { NORMAL: 250, SCRATCH: 250, EDGE: 250, CENTER: 250 },
    },
    stable_across_months: true,
    note: "Lot/die identities are stable across months.",
    cache_coverage: { cached_files: 0, by_month: {} },
  },
};

FIXTURE.all_recommendations = [...FIXTURE.primary_recommendations];

function mockDieApis() {
  const base = FIXTURE.primary_recommendations[0]!;
  vi.spyOn(endpoints, "getCostSavings").mockResolvedValue({
    status: "predicted",
    is_measured_ate_saving: false,
    label: "Predicted DTL Test-Time Cost Saving",
    disclaimer: "Counterfactual estimate — not measured ATE savings.",
    estimator: {
      type: "counterfactual",
      production_facing: true,
      read_only: true,
      mechanism: "M2_adaptive_parametric_condition_pruning",
      label: "Predicted Cost Saving",
      condition_duration_s: 0.05,
      skip_threshold: 0.1,
      tester_cost_per_hour: 25,
      n_baseline_conditions: 4,
      first_condition_id: "COND_RT_NOM",
      cost_source: "configured assumption",
    },
    aggregate: {
      records_evaluated: 108,
      eligible_records: 84,
      records_with_predicted_skip: 70,
      records_with_zero_savings: 38,
      total_baseline_test_time_s: 16.8,
      total_dtl_test_time_s: 6.3,
      total_estimated_seconds_saved: 10.5,
      estimated_seconds_saved_per_record: 0.097,
      predicted_time_saved_pct: 62.5,
      tester_hours_saved: 0.0029,
      tester_cost_per_hour: 25,
      total_predicted_cost_saving: 0.0729,
      predicted_cost_saved_per_record: 0.000675,
      predicted_cost_saved_per_1000_records: 0.675,
      production_volume_supplied: false,
    },
  });
  vi.spyOn(endpoints, "postAnalysisUpload").mockResolvedValue({
    analysis_session_id: "sess-test-1",
    months: ["2026-01", "2026-02", "2026-03"],
    status: "ready",
    used_uploaded_measurements: true,
    used_static_three_month_measurements: false,
    data_provenance: "Analysis generated from uploaded test data",
    primary_die: { lot_id: "DTL_NORM_001", die_id: "DTL_NORM_001_D001" },
  });
  vi.spyOn(endpoints, "getThreeMonthDieRecommendation").mockImplementation(
    async (params) => {
      const found = FIXTURE.primary_recommendations.find(
        (r) =>
          r.production_month === params.production_month &&
          r.parameter_display === params.parameter &&
          r.lot_id === params.lot_id &&
          r.die_id === params.die_id,
      );
      const row = found ?? {
        ...base,
        production_month: params.production_month,
        lot_id: params.lot_id,
        die_id: params.die_id,
        parameter_display: params.parameter,
        recommended_limit:
          params.die_id === "DTL_EDGE_003_D025" ? 60 : base.recommended_limit,
      };
      return {
        recommendation: row,
        candidates: (FIXTURE.candidate_explanations ?? []).filter(
          (c) =>
            c.production_month === params.production_month &&
            c.parameter_display === params.parameter,
        ),
        cached: true,
      };
    },
  );
  vi.spyOn(endpoints, "getThreeMonthDieHistory").mockImplementation(async (params) => {
    const display = params.parameter;
    const history = FIXTURE.primary_recommendations.filter(
      (r) =>
        r.parameter_display === display &&
        r.lot_id === params.lot_id &&
        r.die_id === params.die_id,
    );
    return {
      lot_id: params.lot_id,
      die_id: params.die_id,
      parameter: display,
      parameter_display: display,
      history:
        history.length > 0
          ? history
          : FIXTURE.primary_recommendations.filter((r) => r.parameter_display === "IR_DROP_MV"),
    };
  });
  vi.spyOn(endpoints, "getThreeMonthObserved").mockResolvedValue({
    lot_id: "DTL_NORM_001",
    die_id: "DTL_NORM_001_D001",
    observed_means: {
      IR_DROP_MV: { "2026-01": 23.1, "2026-02": 28.2, "2026-03": 25.3 },
    },
  });
}

describe("analysisDisplay helpers", () => {
  it("finds recommendation without inventing winners", () => {
    const row = findRecommendation(FIXTURE.primary_recommendations, {
      month: "2026-01",
      parameterDisplay: "IR_DROP_MV",
      lotId: "DTL_NORM_001",
      dieId: "DTL_NORM_001_D001",
    });
    expect(row?.recommended_limit).toBe(50);
  });

  it("uses backend why_selected text", () => {
    const row = FIXTURE.primary_recommendations[0];
    expect(whySelectedText(row)).toContain("ML rank tie-break");
  });
});

async function completeUpload() {
  fireEvent.change(screen.getByTestId("upload-january-input"), {
    target: { files: [new File(["a"], "jan.csv", { type: "text/csv" })] },
  });
  fireEvent.change(screen.getByTestId("upload-february-input"), {
    target: { files: [new File(["b"], "feb.csv", { type: "text/csv" })] },
  });
  fireEvent.change(screen.getByTestId("upload-march-input"), {
    target: { files: [new File(["c"], "mar.csv", { type: "text/csv" })] },
  });
  fireEvent.click(screen.getByTestId("upload-analyze"));
  await waitFor(() => expect(screen.getByTestId("upload-provenance")).toBeInTheDocument());
}

describe("ThreeMonthDashboardPage", () => {
  beforeEach(() => {
    vi.spyOn(endpoints, "getThreeMonthAnalysis").mockResolvedValue(FIXTURE);
    mockDieApis();
  });

  it("shows upload prompt before analysis", () => {
    render(<ThreeMonthDashboardPage />);
    expect(screen.getByTestId("upload-prompt")).toHaveTextContent(
      "Upload Jan, Feb and Mar test data to begin analysis.",
    );
    expect(screen.queryByTestId("executive-matrix")).not.toBeInTheDocument();
  });

  it("loads January summary and three-month comparison", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("summary-current")).toBeInTheDocument());
    expect(screen.getByTestId("executive-matrix")).toHaveTextContent(
      "AI RECOMMENDED DYNAMIC TEST LIMITS",
    );
    expect(screen.getByTestId("executive-matrix")).toHaveTextContent("January 2026");
    expect(screen.getByTestId("executive-matrix")).not.toHaveTextContent("GRU");
    expect(screen.queryByText("THREE-MONTH DTL COMPARISON")).not.toBeInTheDocument();
    expect(screen.getByTestId("summary-current")).toHaveTextContent("25 mV");
    expect(screen.getByTestId("summary-recommended")).toHaveTextContent("50 mV");
    expect(screen.getByTestId("summary-decision")).toHaveTextContent("RECOMMEND");
    expect(screen.getByTestId("summary-ml-rank")).toHaveTextContent("#1");
    expect(screen.queryByTestId("summary-model")).not.toBeInTheDocument();
    expect(screen.getByTestId("same-die-three-month-history")).toHaveTextContent("72");
    expect(screen.queryByTestId("synthetic-disclaimer")).not.toBeInTheDocument();
    expect(screen.queryByText(/Decision support only/i)).not.toBeInTheDocument();
    expect(screen.getByTestId("die-hierarchy-selectors")).toBeInTheDocument();
    expect(screen.getByTestId("selected-die-context")).toHaveTextContent("DTL_NORM_001_D001");
    expect(screen.queryByText("Observed Measurement")).not.toBeInTheDocument();
    expect(screen.queryByText("Measurement Distribution")).not.toBeInTheDocument();
    expect(screen.queryByTestId("model-panel")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("cost-savings-card")).toBeInTheDocument());
    expect(screen.getByTestId("cost-savings-disclaimer")).toHaveTextContent(
      "Counterfactual estimate — not measured ATE savings.",
    );
    expect(screen.getByTestId("cost-savings-skips")).toBeInTheDocument();
    expect(screen.getByTestId("upload-provenance")).toHaveTextContent(
      "Analysis generated from uploaded test data",
    );
  });

  it("places AI recommended DTL matrix above selectors", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("executive-matrix")).toBeInTheDocument());
    const dash = screen.getByTestId("three-month-dashboard");
    const matrix = screen.getByTestId("executive-matrix");
    const monthSel = screen.getByTestId("month-selector");
    expect(dash.compareDocumentPosition(matrix) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(matrix.compareDocumentPosition(monthSel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("matrix parameter click selects IR_DROP_MV and reuses engineering detail", async () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;

    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("top-summary")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("matrix-param-THERMAL_C"));
    await waitFor(() => expect(screen.getByTestId("parameter-select")).toHaveValue("THERMAL_C"));

    fireEvent.click(screen.getByTestId("matrix-param-IR_DROP_MV"));
    await waitFor(() => expect(screen.getByTestId("parameter-select")).toHaveValue("IR_DROP_MV"));
    await waitFor(() =>
      expect(screen.getByTestId("hierarchy-parameter-select")).toHaveValue("IR_DROP_MV"),
    );
    await waitFor(() => expect(screen.getByTestId("top-summary")).toBeInTheDocument());
    expect(screen.getByTestId("summary-entity")).toHaveTextContent("IR_DROP_MV");
    expect(screen.getByTestId("summary-recommended")).toHaveTextContent("50 mV");
    expect(screen.getByTestId("why-selected")).toBeInTheDocument();
    expect(screen.getByTestId("candidate-comparison")).toBeInTheDocument();
    expect(screen.getByTestId("same-die-three-month-history")).toBeInTheDocument();
    expect(screen.getAllByTestId("engineering-detail")).toHaveLength(1);
    expect(screen.getAllByTestId("top-summary")).toHaveLength(1);
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it("matrix parameter click selects THERMAL_C and updates engineering detail", async () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;

    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("top-summary")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("matrix-param-THERMAL_C"));
    await waitFor(() => expect(screen.getByTestId("parameter-select")).toHaveValue("THERMAL_C"));
    await waitFor(() =>
      expect(screen.getByTestId("hierarchy-parameter-select")).toHaveValue("THERMAL_C"),
    );
    await waitFor(() => expect(screen.getByTestId("summary-entity")).toHaveTextContent("THERMAL_C"));
    expect(screen.getByTestId("summary-recommended")).toHaveTextContent("92");
    expect(screen.getByTestId("summary-decision")).toHaveTextContent("KEEP_CURRENT");
    expect(screen.getAllByTestId("engineering-detail")).toHaveLength(1);
    expect(screen.getAllByTestId("top-summary")).toHaveLength(1);
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it("switches months from fixture data", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("month-2026-02")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("month-2026-02"));
    await waitFor(() =>
      expect(screen.getByTestId("summary-recommended")).toHaveTextContent("72 mV"),
    );
  });

  it("supports category lot die hierarchy", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("category-select")).toBeInTheDocument());
    fireEvent.change(screen.getByTestId("category-select"), { target: { value: "EDGE" } });
    await waitFor(() => expect(screen.getByTestId("lot-select")).toHaveValue("DTL_EDGE_003"));
    fireEvent.change(screen.getByTestId("die-select"), {
      target: { value: "DTL_EDGE_003_D025" },
    });
    await waitFor(() =>
      expect(screen.getByTestId("selected-die-context")).toHaveTextContent("DTL_EDGE_003_D025"),
    );
    await waitFor(() =>
      expect(screen.getByTestId("summary-recommended")).toHaveTextContent("60"),
    );
  });

  it("shows same-die three-month history panel", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() =>
      expect(screen.getByTestId("same-die-three-month-history")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("same-die-three-month-history")).toHaveTextContent(
      "DTL_NORM_001_D001",
    );
  });

  it("shows generating recommendation context while loading", async () => {
    let resolveRec!: (value: {
      recommendation: (typeof FIXTURE.primary_recommendations)[0];
      candidates: [];
      cached: boolean;
    }) => void;
    vi.spyOn(endpoints, "getThreeMonthDieRecommendation").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRec = resolve as typeof resolveRec;
        }),
    );
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("die-loading")).toBeInTheDocument());
    expect(screen.getByTestId("die-loading")).toHaveTextContent("Generating recommendation...");
    expect(screen.getByTestId("die-loading")).toHaveTextContent("January 2026");
    expect(screen.getByTestId("die-loading")).toHaveTextContent("DTL_NORM_001");
    resolveRec({
      recommendation: FIXTURE.primary_recommendations[0]!,
      candidates: [],
      cached: false,
    });
    await waitFor(() => expect(screen.queryByTestId("die-loading")).not.toBeInTheDocument());
  });

  it("does not render Browse All Dies or Lot Summary", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("three-month-dashboard")).toBeInTheDocument());
    expect(screen.queryByTestId("browse-all-dies")).not.toBeInTheDocument();
    expect(screen.queryByTestId("browse-all-dies-toggle")).not.toBeInTheDocument();
    expect(screen.queryByTestId("lot-summary")).not.toBeInTheDocument();
    expect(screen.queryByTestId("lot-summary-toggle")).not.toBeInTheDocument();
    expect(screen.queryByTestId("three-month-history")).not.toBeInTheDocument();
  });

  it("shows why-selected and candidate comparison", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("why-selected-text")).toBeInTheDocument());
    expect(screen.getByTestId("why-selected-text")).toHaveTextContent("ML rank tie-break");
    expect(screen.getByTestId("candidate-comparison")).toHaveTextContent("50");
    expect(screen.getByTestId("ml-tie-break-card")).toBeInTheDocument();
  });

  it("supports KEEP_CURRENT decision display", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("parameter-select")).toBeInTheDocument());
    fireEvent.change(screen.getByTestId("parameter-select"), {
      target: { value: "THERMAL_C" },
    });
    await waitFor(() =>
      expect(screen.getByTestId("summary-decision")).toHaveTextContent("KEEP_CURRENT"),
    );
  });

  it("shows API error state", async () => {
    vi.spyOn(endpoints, "getThreeMonthAnalysis").mockRejectedValue(
      new Error("API error boom"),
    );
    render(<ThreeMonthDashboardPage />);
    fireEvent.change(screen.getByTestId("upload-january-input"), {
      target: { files: [new File(["a"], "jan.csv", { type: "text/csv" })] },
    });
    fireEvent.change(screen.getByTestId("upload-february-input"), {
      target: { files: [new File(["b"], "feb.csv", { type: "text/csv" })] },
    });
    fireEvent.change(screen.getByTestId("upload-march-input"), {
      target: { files: [new File(["c"], "mar.csv", { type: "text/csv" })] },
    });
    fireEvent.click(screen.getByTestId("upload-analyze"));
    await waitFor(() => expect(screen.getByText(/API error boom/)).toBeInTheDocument());
  });

  it("does not show non-scorable block in UI", async () => {
    render(<ThreeMonthDashboardPage />);
    await completeUpload();
    await waitFor(() => expect(screen.getByTestId("parameter-selector")).toBeInTheDocument());
    expect(screen.getByTestId("parameter-selector")).not.toHaveTextContent("non-scorable");
  });
});
