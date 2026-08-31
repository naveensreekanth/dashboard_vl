"use client";

import { useEffect, useState } from "react";
import { DisconnectedBanner } from "@/components/common/DisconnectedBanner";
import { ErrorState } from "@/components/common/ErrorState";
import { LiveStatusIndicator } from "@/components/common/LiveStatusIndicator";
import { LoadingState } from "@/components/common/LoadingState";
import { VerilumenBrand } from "@/components/branding/VerilumenBrand";
import { SessionControl } from "@/components/auth/SessionControl";
import { EnterpriseControls } from "@/components/dashboard/EnterpriseControls";
import { TestFloorEventLog } from "@/components/events/TestFloorEventLog";
import { HeaderStats } from "@/components/kpi/HeaderStats";
import { OptimizationKpiGrid } from "@/components/kpi/OptimizationKpiGrid";
import { DynamicTestLimits } from "@/components/limits/DynamicTestLimits";
import { PredictiveMaintenanceCard } from "@/components/maintenance/PredictiveMaintenanceCard";
import { UploadControl } from "@/components/uploads/UploadControl";
import { WaferMap } from "@/components/wafer/WaferMap";
import { YieldSummary } from "@/components/wafer/YieldSummary";
import { useQuery } from "@tanstack/react-query";
import { useDashboardData } from "@/hooks/useDashboardData";
import { useWaferRealtime } from "@/hooks/useWaferRealtime";
import { fetchWafer } from "@/services/api";
import { useOpsStore } from "@/stores/opsStore";

export function DashboardShell() {
  const { summary, isLoading, isError, refetch } = useDashboardData();
  const selectedWaferId = useOpsStore((s) => s.waferId);
  const hydrateFromSummary = useOpsStore((s) => s.hydrateFromSummary);

  useEffect(() => {
    if (!summary?.active_wafer) return;
    // Lot/wafer from backend only — never inject default tester/site.
    hydrateFromSummary({
      lotId: summary.active_wafer.lot_id,
      waferId: summary.active_wafer.wafer_id,
      testerId: null,
      siteId: null,
    });
  }, [summary, hydrateFromSummary]);

  const waferId = selectedWaferId || summary?.active_wafer?.wafer_id || null;
  useWaferRealtime(waferId, true);

  const waferQuery = useQuery({
    queryKey: ["wafer", waferId, "detail"],
    queryFn: () => fetchWafer(waferId!),
    enabled: Boolean(waferId),
    staleTime: 8_000,
  });

  if (isLoading && !summary) {
    return (
      <div className="mx-auto max-w-[1400px] px-7 pb-[90px] pt-[30px]">
        <LoadingState />
      </div>
    );
  }

  if (isError && !summary) {
    return (
      <div className="mx-auto max-w-[1400px] px-7 pb-[90px] pt-[30px]">
        <ErrorState
          message="Unable to load dashboard summary from the API."
          onRetry={() => void refetch()}
        />
      </div>
    );
  }

  const wafer = waferQuery.data ?? summary?.active_wafer ?? null;
  const [activeTab, setActiveTab] = useState<"overview" | "shmoo" | "test_time" | "dtl" | "retest">("overview");

  useEffect(() => {
    const handler = (e: Event) => {
      const custom = e as CustomEvent<{ tab: "overview" | "shmoo" | "test_time" | "dtl" | "retest" }>;
      if (custom.detail?.tab) setActiveTab(custom.detail.tab);
    };
    window.addEventListener("switchSuiteTab", handler);
    return () => window.removeEventListener("switchSuiteTab", handler);
  }, []);

  return (
    <div className="mx-auto max-w-[1440px] px-7 pb-[90px] pt-[24px]">
      <DisconnectedBanner />

      <header className="vl-header vl-section-enter relative z-50 mb-[20px] flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-6">
          <VerilumenBrand size="header" />
          <nav className="flex items-center gap-1 rounded-lg border border-[var(--line)] bg-[rgba(10,18,30,0.85)] p-1 shadow-inner backdrop-blur-md">
            <button
              onClick={() => setActiveTab("overview")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11.5px] font-semibold transition-all ${
                activeTab === "overview"
                  ? "bg-[rgba(107,193,242,0.18)] text-[var(--cyan)] shadow-[0_0_12px_rgba(107,193,242,0.25)] border border-[rgba(107,193,242,0.35)]"
                  : "text-[var(--muted)] hover:text-white"
              }`}
            >
              🌐 Overview
            </button>
            <button
              onClick={() => setActiveTab("shmoo")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11.5px] font-semibold transition-all ${
                activeTab === "shmoo"
                  ? "bg-[rgba(107,193,242,0.18)] text-[var(--cyan)] shadow-[0_0_12px_rgba(107,193,242,0.25)] border border-[rgba(107,193,242,0.35)]"
                  : "text-[var(--muted)] hover:text-white"
              }`}
            >
              ⚡ M-BIST Shmoo
            </button>
            <button
              onClick={() => setActiveTab("test_time")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11.5px] font-semibold transition-all ${
                activeTab === "test_time"
                  ? "bg-[rgba(107,193,242,0.18)] text-[var(--cyan)] shadow-[0_0_12px_rgba(107,193,242,0.25)] border border-[rgba(107,193,242,0.35)]"
                  : "text-[var(--muted)] hover:text-white"
              }`}
            >
              ⏱️ Vector Memory
            </button>
            <button
              onClick={() => setActiveTab("dtl")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11.5px] font-semibold transition-all ${
                activeTab === "dtl"
                  ? "bg-[rgba(107,193,242,0.18)] text-[var(--cyan)] shadow-[0_0_12px_rgba(107,193,242,0.25)] border border-[rgba(107,193,242,0.35)]"
                  : "text-[var(--muted)] hover:text-white"
              }`}
            >
              📊 Dynamic Limits (DTL)
            </button>
            <button
              onClick={() => setActiveTab("retest")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11.5px] font-semibold transition-all ${
                activeTab === "retest"
                  ? "bg-[rgba(107,193,242,0.18)] text-[var(--cyan)] shadow-[0_0_12px_rgba(107,193,242,0.25)] border border-[rgba(107,193,242,0.35)]"
                  : "text-[var(--muted)] hover:text-white"
              }`}
            >
              🎯 Retest AI
            </button>
          </nav>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <HeaderStats data={summary?.header ?? null} />
          <div className="flex items-center gap-3 border-l border-[var(--line)] pl-4">
            <UploadControl />
            <div className="text-right">
              <div className="vl-label mb-0.5">Live Connection</div>
              <LiveStatusIndicator />
            </div>
            <SessionControl />
          </div>
        </div>
      </header>

      {activeTab === "overview" ? (
        <>
          <div className="vl-section-enter vl-section-enter-delay-1">
            <EnterpriseControls />
          </div>

          <section
            id="live-wafer-map"
            className="vl-surface-deep vl-section-enter vl-section-enter-delay-2 mb-[30px] grid grid-cols-1 gap-[26px] p-6 md:grid-cols-[340px_1fr]"
          >
            <WaferMap waferId={wafer?.wafer_id ?? null} />
            <YieldSummary wafer={wafer} />
          </section>

          <div id="optimization-parameters" className="vl-section-title mb-3 scroll-mt-6 vl-section-enter vl-section-enter-delay-3">
            Optimization Parameters
          </div>
          <div className="vl-section-enter vl-section-enter-delay-3">
            <OptimizationKpiGrid>
              <PredictiveMaintenanceCard />
              <DynamicTestLimits data={summary?.test_limits ?? null} />
            </OptimizationKpiGrid>
          </div>

          <div className="mt-1 vl-section-enter vl-section-enter-delay-4">
            <TestFloorEventLog />
          </div>
        </>
      ) : (
        <div className="vl-surface-deep relative flex flex-col overflow-hidden rounded-xl border border-[var(--line)] p-4 shadow-2xl">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--line)] pb-3">
            <div className="flex items-center gap-2.5">
              <span className="text-[14px] font-bold tracking-wide text-white">
                {activeTab === "shmoo" && "⚡ M-BIST Shmoo ML Optimization Engine"}
                {activeTab === "test_time" && "⏱️ ATE Vector Memory & Test Time Optimization"}
                {activeTab === "dtl" && "📊 Dynamic Test Limits (DTL) Recommendation Engine"}
                {activeTab === "retest" && "🎯 AI Retest-Benefit Prediction Intelligence"}
              </span>
              <span className="rounded bg-[var(--green-dim)] px-2 py-0.5 text-[10px] font-bold text-[var(--green)]">
                OFFLINE INSTANCE
              </span>
            </div>
            <button
              onClick={() => setActiveTab("overview")}
              className="rounded-md border border-[var(--line)] bg-[rgba(255,255,255,0.06)] px-3 py-1.5 text-[11.5px] font-semibold text-[var(--cyan)] transition-colors hover:bg-[rgba(107,193,242,0.15)]"
            >
              ← Back to Central Overview
            </button>
          </div>
          <iframe
            src={
              activeTab === "shmoo"
                ? "http://127.0.0.1:5000"
                : activeTab === "test_time"
                ? "http://127.0.0.1:5173"
                : activeTab === "dtl"
                ? "http://127.0.0.1:5174/three-month"
                : "http://127.0.0.1:5175"
            }
            className="h-[calc(100vh-210px)] min-h-[640px] w-full rounded-lg border-0 bg-transparent"
            title="Embedded Suite Microservice"
          />
        </div>
      )}

      <footer className="mt-8 flex flex-wrap justify-between gap-2 border-t border-[rgba(107,193,242,0.18)] pt-4 text-[11px] text-[#7f96b0] vl-section-enter vl-section-enter-delay-5">
        <span>
          Metrics reflect an ML-assisted test-optimization layer over standard ATE limits and bin
          logic
        </span>
        <span className="font-mono text-[10px] tracking-wide text-[#9ec9ef]">
          {summary?.connection_hint ??
            "Live telemetry · PostgreSQL · Redis · authenticated WebSocket"}
        </span>
      </footer>
    </div>
  );
}
