"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { fetchDashboardSummary } from "@/services/api";
import { useConnectionStore } from "@/stores/connectionStore";
import { useDashboardStore } from "@/stores/dashboardStore";

/** Dashboard chrome data. Wafer die map has its own hooks (useWafer / useWaferDies / useWaferRealtime). */
export function useDashboardData() {
  const setSummary = useDashboardStore((s) => s.setSummary);
  const status = useConnectionStore((s) => s.status);
  const frozen = status === "STALE" || status === "OFFLINE" || status === "RECONNECTING";

  const summaryQuery = useQuery({
    queryKey: ["dashboard", "summary"],
    queryFn: fetchDashboardSummary,
    refetchInterval: frozen ? false : 15_000,
    refetchOnWindowFocus: !frozen,
  });

  useEffect(() => {
    if (frozen) return;
    if (summaryQuery.data) setSummary(summaryQuery.data);
  }, [summaryQuery.data, setSummary, frozen]);

  return {
    summary: summaryQuery.data ?? null,
    isLoading: summaryQuery.isLoading,
    isError: summaryQuery.isError,
    error: summaryQuery.error,
    refetch: async () => {
      await summaryQuery.refetch();
    },
  };
}
