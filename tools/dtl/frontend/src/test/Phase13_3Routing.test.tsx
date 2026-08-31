import { render, screen } from "@testing-library/react";
import { MemoryRouter, Navigate, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { DashboardLayout } from "@/components/layout/DashboardLayout";

function ThreeMonthStub() {
  return <div data-testid="three-month-dashboard">Three-Month DTL Recommendation</div>;
}

function AppRoutes() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route index element={<Navigate to="/three-month" replace />} />
        <Route path="three-month" element={<ThreeMonthStub />} />
        <Route path="*" element={<Navigate to="/three-month" replace />} />
      </Route>
    </Routes>
  );
}

describe("Phase 13.3 routing", () => {
  it("opens Three-Month from root /", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <AppRoutes />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("three-month-dashboard")).toBeInTheDocument();
    expect(screen.getByText("Three-Month Recommendation")).toBeInTheDocument();
    expect(screen.queryByText(/Decision support only/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Overview")).not.toBeInTheDocument();
    expect(screen.queryByText("Audit / Evidence")).not.toBeInTheDocument();
    expect(screen.queryByText("About")).not.toBeInTheDocument();
    expect(document.body.textContent ?? "").not.toMatch(/GRU/i);
  });

  it("keeps /three-month as the recommendation page", () => {
    render(
      <MemoryRouter initialEntries={["/three-month"]}>
        <AppRoutes />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("three-month-dashboard")).toBeInTheDocument();
  });
});
