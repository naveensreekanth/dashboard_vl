"use client";

import { KpiMetricTile, type KpiMetricTileProps } from "@/components/kpi/KpiMetricTile";
import {
  SHMOO_CAPABILITIES,
  type ShmooCapabilityMetric,
} from "@/lib/kpiExternalPages";

export function toShmooTileProps(
  metrics?: ShmooCapabilityMetric[],
  size: "card" | "popup" = "card",
): KpiMetricTileProps[] {
  return SHMOO_CAPABILITIES.map((cap) => {
    const m = metrics?.find((x) => x.id === cap.id);
    return {
      eyebrow: "SHMOO Capability",
      title: cap.label,
      variant: "cyan" as const,
      size,
      metrics: [
        {
          label: m?.primaryLabel ?? cap.primaryLabel,
          value: m?.value ?? Number.NaN,
          unit: m?.unit ?? "%",
        },
        {
          label: m?.secondaryLabel ?? cap.secondaryLabel,
          value: m?.secondaryValue ?? cap.secondaryValue,
          unit: m?.secondaryUnit ?? cap.secondaryUnit,
        },
      ],
    };
  });
}

export function ShmooCapabilityTiles({
  metrics,
  size = "card",
}: {
  metrics?: ShmooCapabilityMetric[];
  size?: "card" | "popup";
}) {
  return (
    <>
      {toShmooTileProps(metrics, size).map((props, i) => (
        <KpiMetricTile key={SHMOO_CAPABILITIES[i].id} {...props} />
      ))}
    </>
  );
}
