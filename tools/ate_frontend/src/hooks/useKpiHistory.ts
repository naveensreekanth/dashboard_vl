"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchKpiHistory } from "@/services/api";

export function useKpiHistory(kpiId: string | null | undefined, limit = 48) {
  const query = useQuery({
    queryKey: ["kpis", kpiId, "history", limit],
    queryFn: () => fetchKpiHistory(kpiId!, limit),
    enabled: Boolean(kpiId),
    staleTime: 8_000,
  });

  return {
    history: query.data?.history ?? [],
    unit: query.data?.unit,
    name: query.data?.name,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}
