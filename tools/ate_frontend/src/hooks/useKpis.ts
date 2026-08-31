"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { fetchKpi, fetchKpis } from "@/services/api";
import { useConnectionStore } from "@/stores/connectionStore";
import { useKpiStore } from "@/stores/kpiStore";
import type { Kpi } from "@/types/kpi";

/**
 * Loads optimization KPIs from the authoritative backend.
 * When telemetry is STALE/OFFLINE — keep last hydrated values; never invent replacements.
 */
export function useKpis() {
  const hydrateKpis = useKpiStore((s) => s.hydrateKpis);
  const kpisById = useKpiStore((s) => s.kpisById);
  const status = useConnectionStore((s) => s.status);
  const frozen = status === "STALE" || status === "OFFLINE" || status === "RECONNECTING";

  const query = useQuery({
    queryKey: ["kpis"],
    queryFn: fetchKpis,
    staleTime: 8_000,
    // Do not poll-replace KPIs while the floor stream is unhealthy
    refetchInterval: frozen ? false : 15_000,
    refetchOnWindowFocus: !frozen,
  });

  useEffect(() => {
    if (frozen) return;
    if (query.data?.kpis) {
      hydrateKpis(query.data.kpis);
    }
  }, [query.data, hydrateKpis, frozen]);

  const kpis: Kpi[] = Object.keys(kpisById).length
    ? Object.values(kpisById)
    : (query.data?.kpis ?? []);

  return {
    kpis,
    isLoading: query.isLoading && !Object.keys(kpisById).length,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}

export function useKpiDetail(kpiId: string | null) {
  return useQuery({
    queryKey: ["kpis", kpiId],
    queryFn: () => fetchKpi(kpiId!),
    enabled: Boolean(kpiId),
  });
}
