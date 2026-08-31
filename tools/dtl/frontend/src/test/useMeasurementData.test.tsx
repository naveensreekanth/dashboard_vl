import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import type { ReactNode } from "react";
import type {
  DieConditionsResponse,
  DieDistributionResponse,
  DieMeasurementResponse,
} from "@/api/types";
import { AppProvider } from "@/state/AppProvider";
import { useAppContext } from "@/state/useAppContext";
import { useMeasurementData } from "@/hooks/useMeasurementData";

vi.mock("@/api/endpoints", () => ({
  getDieMeasurements: vi.fn(),
  getDieDistribution: vi.fn(),
  getDieConditions: vi.fn(),
}));

import {
  getDieConditions,
  getDieDistribution,
  getDieMeasurements,
} from "@/api/endpoints";

const getDieMeasurementsMock = vi.mocked(getDieMeasurements);
const getDieDistributionMock = vi.mocked(getDieDistribution);
const getDieConditionsMock = vi.mocked(getDieConditions);

function wrapper({ children }: { children: ReactNode }) {
  return <AppProvider>{children}</AppProvider>;
}

const distFixture: DieDistributionResponse = {
  lot_id: "L1",
  die_id: "D1",
  parameter: "ir_drop",
  domain: "core",
  unit: "mV",
  scope: "die",
  n: 200,
  min: 1,
  median: 2,
  p95: 3,
  max: 4,
  source_classification: "SYNTHETIC",
  dataset_version: "V1",
  stats_method: "phase3_compute_dist_stats",
  found: true,
};

const condFixture: DieConditionsResponse = {
  lot_id: "L1",
  die_id: "D1",
  parameter: "ir_drop",
  domain: "core",
  unit: "mV",
  source_classification: "SYNTHETIC",
  dataset_version: "V1",
  found: false,
  reason: "not_condition_aware",
  conditions: [],
};

describe("useMeasurementData race protection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getDieDistributionMock.mockResolvedValue(distFixture);
    getDieConditionsMock.mockResolvedValue(condFixture);
  });

  it("ignores stale measurement responses after selection change", async () => {
    let resolveFirst!: (v: DieMeasurementResponse) => void;
    getDieMeasurementsMock
      .mockImplementationOnce(
        () =>
          new Promise<DieMeasurementResponse>((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockResolvedValueOnce({
        lot_id: "L1",
        die_id: "D2",
        parameter: "ir_drop",
        domain: "core",
        unit: "mV",
        observed_value: 99,
        observed_value_rule: "median_over_patterns",
        source_classification: "SYNTHETIC",
        dataset_version: "V1",
        found: true,
      });

    const { result } = renderHook(
      () => {
        const ctx = useAppContext();
        const meas = useMeasurementData();
        return { ctx, meas };
      },
      { wrapper },
    );

    act(() => {
      result.current.ctx.dispatch({ type: "SET_LOT", payload: "L1" });
    });
    act(() => {
      result.current.ctx.dispatch({ type: "SET_DIE", payload: "D1" });
    });
    act(() => {
      result.current.ctx.dispatch({ type: "SET_PARAMETER", payload: "ir_drop" });
    });

    await waitFor(() => {
      expect(getDieMeasurementsMock).toHaveBeenCalledTimes(1);
    });

    act(() => {
      result.current.ctx.dispatch({ type: "SET_DIE", payload: "D2" });
    });
    act(() => {
      result.current.ctx.dispatch({ type: "SET_PARAMETER", payload: "ir_drop" });
    });

    act(() => {
      resolveFirst({
        lot_id: "L1",
        die_id: "D1",
        parameter: "ir_drop",
        domain: "core",
        unit: "mV",
        observed_value: 11.11,
        observed_value_rule: "median_over_patterns",
        source_classification: "SYNTHETIC",
        dataset_version: "V1",
        found: true,
      });
    });

    await waitFor(() => {
      expect(result.current.meas.measurement?.die_id).toBe("D2");
      expect(result.current.meas.measurement?.observed_value).toBe(99);
    });
    expect(result.current.meas.measurement?.observed_value).not.toBe(11.11);
  });
});
