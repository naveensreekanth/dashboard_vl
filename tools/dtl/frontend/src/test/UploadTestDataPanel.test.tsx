import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UploadAnalysisPanel } from "@/components/threeMonth/UploadTestDataPanel";
import * as endpoints from "@/api/endpoints";

describe("UploadAnalysisPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("requires all three files before analyze is enabled", () => {
    render(<UploadAnalysisPanel onSessionReady={vi.fn()} />);
    expect(screen.getByTestId("upload-analysis-panel")).toHaveTextContent("Upload DTL Test Data");
    expect(screen.getByTestId("upload-analyze")).toBeDisabled();

    const jan = new File(["a"], "jan.csv", { type: "text/csv" });
    fireEvent.change(screen.getByTestId("upload-january-input"), { target: { files: [jan] } });
    expect(screen.getByTestId("upload-analyze")).toBeDisabled();

    const feb = new File(["b"], "feb.csv", { type: "text/csv" });
    const mar = new File(["c"], "mar.csv", { type: "text/csv" });
    fireEvent.change(screen.getByTestId("upload-february-input"), { target: { files: [feb] } });
    fireEvent.change(screen.getByTestId("upload-march-input"), { target: { files: [mar] } });
    expect(screen.getByTestId("upload-analyze")).not.toBeDisabled();
  });

  it("calls onSessionReady after successful upload", async () => {
    const onReady = vi.fn();
    vi.spyOn(endpoints, "postAnalysisUpload").mockResolvedValue({
      analysis_session_id: "sess-1",
      months: ["2026-01", "2026-02", "2026-03"],
      status: "ready",
      used_uploaded_measurements: true,
      used_static_three_month_measurements: false,
      data_provenance: "Analysis generated from uploaded test data",
      primary_die: { lot_id: "DTL_NORM_001", die_id: "DTL_NORM_001_D001" },
    });

    render(<UploadAnalysisPanel onSessionReady={onReady} />);
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
    await waitFor(() => expect(onReady).toHaveBeenCalled());
    expect(onReady.mock.calls[0]?.[0]?.analysis_session_id).toBe("sess-1");
  });

  it("shows error state on failure", async () => {
    vi.spyOn(endpoints, "postAnalysisUpload").mockRejectedValue(new Error("bad file"));
    render(<UploadAnalysisPanel onSessionReady={vi.fn()} />);
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
    await waitFor(() =>
      expect(screen.getByTestId("upload-analysis-error")).toHaveTextContent("bad file"),
    );
  });
});
