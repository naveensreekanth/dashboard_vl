import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useTheme } from "@/state/ThemeContext";

const navSections = [
  {
    title: "TEST LIMITS",
    items: [
      { to: "/three-month", label: "Three-Month Recommendation", exact: true },
    ],
  },
];

export function DashboardLayout() {
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-[var(--bg-app)] text-[var(--text-primary)] transition-colors duration-150">
      {/* Engineering Navigation Sidebar */}
      <aside className="w-full md:w-64 shrink-0 border-r border-[var(--border-subtle)] bg-[var(--bg-panel)] flex flex-col justify-between">
        <div>
          {/* Brand Header */}
          <div className="p-4 border-b border-[var(--border-subtle)]">
            <div className="flex items-center gap-2">
              <div className="h-6 w-1.5 rounded-sm bg-cyan-500" />
              <div>
                <h1 className="text-xs font-bold tracking-widest text-[var(--text-primary)] uppercase">
                  DTL Recommendation
                </h1>
                <p className="text-[10px] text-[var(--text-muted)] font-mono">
                  ATE Limit Optimization
                </p>
              </div>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="p-3 space-y-4" aria-label="Main navigation">
            {navSections.map((sec) => (
              <div key={sec.title} className="space-y-1">
                <p className="px-3 text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">
                  {sec.title}
                </p>
                {sec.items.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={({ isActive }) =>
                      `flex items-center gap-2.5 rounded-md px-3 py-2 text-xs font-medium transition-colors ${
                        isActive
                          ? "bg-[var(--accent-subtle)] text-[var(--accent)] border-l-2 border-[var(--accent)] font-semibold"
                          : "text-[var(--text-secondary)] hover:bg-[var(--bg-panel-secondary)] hover:text-[var(--text-primary)]"
                      }`
                    }
                  >
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </div>
            ))}
          </nav>
        </div>

        {/* Sidebar Footer — System Info */}
        <div className="p-3 border-t border-[var(--border-subtle)] text-[11px] text-[var(--text-muted)] font-mono space-y-1 bg-[var(--bg-panel-secondary)]">
          <div className="flex items-center justify-between">
            <span>Status</span>
            <span className="inline-flex items-center gap-1 text-emerald-500 font-semibold text-[10px]">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              READY
            </span>
          </div>
          <div className="flex items-center justify-between text-[10px]">
            <span>Engine</span>
            <span>v1.0-release</span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-12 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center text-xs font-mono">
            <span className="text-[var(--text-primary)] font-medium">
              {location.pathname.includes("three-month")
                ? "Three-Month DTL Recommendation"
                : "DTL Dashboard"}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Theme Toggle Button */}
            <button
              type="button"
              onClick={toggleTheme}
              className="inline-flex items-center gap-1.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)] px-2.5 py-1 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-muted)] transition-colors focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              data-testid="theme-toggle"
            >
              {theme === "dark" ? (
                <>
                  <span aria-hidden="true" className="text-amber-400">☀</span>
                  <span className="font-sans">Light</span>
                </>
              ) : (
                <>
                  <span aria-hidden="true" className="text-cyan-600">☾</span>
                  <span className="font-sans">Dark</span>
                </>
              )}
            </button>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
