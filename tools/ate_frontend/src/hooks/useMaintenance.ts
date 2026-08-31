"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchMaintenance, fetchMaintenanceTester } from "@/services/api";

/** Loads predictive maintenance projections from the Python ML service. */
export function useMaintenance() {
  const query = useQuery({
    queryKey: ["maintenance"],
    queryFn: fetchMaintenance,
    staleTime: 8_000,
  });

  return {
    data: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
  };
}

export function useMaintenanceTester(testerId: string | null) {
  return useQuery({
    queryKey: ["maintenance", testerId],
    queryFn: () => fetchMaintenanceTester(testerId!),
    enabled: Boolean(testerId),
  });
}
