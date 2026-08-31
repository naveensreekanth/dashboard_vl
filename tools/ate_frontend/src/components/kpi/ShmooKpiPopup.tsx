"use client";

import { ExternalToolPopup } from "@/components/kpi/ExternalToolPopup";
import { ShmooCapabilityTiles } from "@/components/kpi/ShmooCapabilityTiles";
import { ShmooPlotPreview } from "@/components/kpi/ShmooPlotPreview";
import { SHMOO_VL_BASE, type ShmooCapabilityMetric } from "@/lib/kpiExternalPages";

/**
 * SHMOO ML-Based Optimization popup — tiles + staged plot preview + open tool.
 */
export function ShmooKpiPopup({
  title,
  onClose,
  metrics,
}: {
  title: string;
  onClose: () => void;
  metrics?: ShmooCapabilityMetric[];
}) {
  return (
    <ExternalToolPopup
      title={title}
      onClose={onClose}
      description="The SHMOO ML Optimization tool runs on a separate page. Open it in a new tab to upload datasets, run boundary prediction, and view plots."
      ctaLabel="Open SHMOO ML Tool"
      ctaHref={SHMOO_VL_BASE}
    >
      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-1 gap-x-6 gap-y-3 border-t border-[rgba(107,193,242,0.14)] pt-3 sm:grid-cols-2">
          <ShmooCapabilityTiles metrics={metrics} size="popup" />
        </div>
        <ShmooPlotPreview />
      </div>
    </ExternalToolPopup>
  );
}
