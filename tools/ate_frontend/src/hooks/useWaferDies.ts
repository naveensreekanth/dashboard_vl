"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";
import { mapDieOut } from "@/lib/waferMappers";
import { fetchWaferDies } from "@/services/api";
import { useWaferStore } from "@/stores/waferStore";
import type { Die } from "@/types/wafer";

export function useWaferDies(waferId: string | null | undefined) {
  const hydrateDies = useWaferStore((s) => s.hydrateDies);
  const diesById = useWaferStore((s) => s.diesById);
  const setLifecycle = useWaferStore((s) => s.setLifecycle);

  const query = useQuery({
    queryKey: ["wafer", waferId, "dies"],
    queryFn: () => fetchWaferDies(waferId!),
    enabled: Boolean(waferId),
    staleTime: 5_000,
    refetchOnMount: "always",
  });

  useEffect(() => {
    if (!waferId || !query.data) return;
    hydrateDies(query.data.map(mapDieOut));
  }, [query.data, waferId, hydrateDies]);

  useEffect(() => {
    if (query.isError) setLifecycle("error");
    else if (query.isSuccess && query.data.length === 0) setLifecycle("empty");
  }, [query.isError, query.isSuccess, query.data, setLifecycle]);

  const dies: Die[] = useMemo(() => {
    if (query.data?.length && waferId) {
      return query.data.map(mapDieOut).filter((d) => d.wafer_id === waferId);
    }
    if (!waferId) return [];
    return Object.values(diesById).filter((d) => d.wafer_id === waferId);
  }, [query.data, diesById, waferId]);

  return {
    dies,
    diesById,
    isLoading: query.isLoading || query.isFetching,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}
