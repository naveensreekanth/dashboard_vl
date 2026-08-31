import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { AboutPage } from "@/pages/AboutPage";
import { AuditPage } from "@/pages/AuditPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { RecommendationPage } from "@/pages/RecommendationPage";
import { ThreeMonthDashboardPage } from "@/pages/ThreeMonthDashboardPage";
import { AppProvider } from "@/state/AppProvider";
import { ThemeProvider } from "@/state/ThemeContext";

/**
 * Phase 13.3 — Three-Month is the canonical landing experience.
 * Legacy Overview / Recommendation / Audit / About routes remain registered
 * (not linked in nav) so deep links do not 404 during transition.
 */
export default function App() {
  return (
    <ThemeProvider>
      <AppProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<DashboardLayout />}>
              <Route index element={<Navigate to="/three-month" replace />} />
              <Route path="three-month" element={<ThreeMonthDashboardPage />} />
              {/* Legacy routes — hidden from navigation */}
              <Route path="overview" element={<OverviewPage />} />
              <Route path="recommendation" element={<RecommendationPage />} />
              <Route path="audit" element={<AuditPage />} />
              <Route path="about" element={<AboutPage />} />
              <Route path="*" element={<Navigate to="/three-month" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AppProvider>
    </ThemeProvider>
  );
}
