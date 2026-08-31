"use client";

import { useEffect } from "react";
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

  return (
    <div className="mx-auto max-w-[1400px] px-7 pb-[90px] pt-[30px]">
      <DisconnectedBanner />

      <header className="vl-header vl-section-enter relative z-50 mb-[28px] flex flex-wrap items-end justify-between gap-6">
        <div>
          <VerilumenBrand size="header" />
        </div>
        <div className="flex flex-wrap items-end gap-4">
          <HeaderStats data={summary?.header ?? null} />
          <div className="flex items-end gap-3 border-l border-[var(--line)] pl-4">
            <UploadControl />
            <div className="text-right">
              <div className="vl-label mb-1">Live Connection</div>
              <LiveStatusIndicator />
            </div>
            <SessionControl />
          </div>
        </div>
      </header>

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
