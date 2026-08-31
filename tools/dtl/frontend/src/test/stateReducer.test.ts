import { describe, expect, it } from "vitest";
import { appReducer, initialState } from "@/state/types";

describe("state reducer selector behavior", () => {
  it("resets die and parameter when lot changes", () => {
    const withSelections = {
      ...initialState,
      selectedLot: "L1",
      selectedDie: "X1",
      selectedParameter: "ir_drop",
      availableDies: ["X1"],
      availableParameters: ["ir_drop"],
    };
    const next = appReducer(withSelections, { type: "SET_LOT", payload: "L2" });
    expect(next.selectedLot).toBe("L2");
    expect(next.selectedDie).toBe("");
    expect(next.selectedParameter).toBe("");
    expect(next.availableDies).toEqual([]);
    expect(next.availableParameters).toEqual([]);
  });

  it("resets parameter when die changes", () => {
    const withSelections = {
      ...initialState,
      selectedLot: "L1",
      selectedDie: "X1",
      selectedParameter: "VMIN",
      availableParameters: ["VMIN"],
    };
    const next = appReducer(withSelections, { type: "SET_DIE", payload: "X2" });
    expect(next.selectedDie).toBe("X2");
    expect(next.selectedParameter).toBe("");
    expect(next.availableParameters).toEqual([]);
  });
});
