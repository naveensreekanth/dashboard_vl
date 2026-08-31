import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, beforeEach } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { ThemeProvider, useTheme } from "@/state/ThemeContext";

function TestThemeConsumer() {
  const { theme } = useTheme();
  return <div data-testid="current-theme-val">{theme}</div>;
}

describe("ThemeContext & Toggle", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.className = "";
  });

  it("defaults to dark theme when no preference in localStorage", () => {
    render(
      <ThemeProvider>
        <TestThemeConsumer />
      </ThemeProvider>
    );
    expect(screen.getByTestId("current-theme-val")).toHaveTextContent("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("toggles theme immediately and persists to localStorage", () => {
    render(
      <ThemeProvider>
        <MemoryRouter>
          <DashboardLayout />
        </MemoryRouter>
      </ThemeProvider>
    );

    const toggleBtn = screen.getByTestId("theme-toggle");
    expect(toggleBtn).toBeInTheDocument();
    expect(document.documentElement.classList.contains("dark")).toBe(true);

    // Click to switch to light
    fireEvent.click(toggleBtn);
    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(localStorage.getItem("dtl-theme")).toBe("light");

    // Click to switch back to dark
    fireEvent.click(toggleBtn);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.classList.contains("light")).toBe(false);
    expect(localStorage.getItem("dtl-theme")).toBe("dark");
  });

  it("restores previously saved theme from localStorage", () => {
    localStorage.setItem("dtl-theme", "light");
    render(
      <ThemeProvider>
        <TestThemeConsumer />
      </ThemeProvider>
    );
    expect(screen.getByTestId("current-theme-val")).toHaveTextContent("light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
  });
});
