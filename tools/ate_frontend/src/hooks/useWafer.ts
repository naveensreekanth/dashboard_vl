"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { mapWaferDetail } from "@/lib/waferMappers";
import { fetchWafer } from "@/services/api";
import { useWaferStore } from "@/stores/waferStore";

export function useWafer(waferId: string | null | undefined) {
  const hydrateWafer = useWaferStore((s) => s.hydrateWafer);
  const setWaferId = useWaferStore((s) => s.setWaferId);
  const setLifecycle = useWaferStore((s) => s.setLifecycle);
  const wafer = useWaferStore((s) => s.wafer);
  const lifecycle = useWaferStore((s) => s.lifecycle);

  useEffect(() => {
    setWaferId(waferId ?? null);
  }, [waferId, setWaferId]);

  const query = useQuery({
    queryKey: ["wafer", waferId],
    queryFn: () => fetchWafer(waferId!),
    enabled: Boolean(waferId),
    staleTime: 10_000,
  });

  useEffect(() => {
    if (query.data) {
      hydrateWafer(mapWaferDetail(query.data));
    }
  }, [query.data, hydrateWafer]);

  useEffect(() => {
    if (query.isLoading) setLifecycle("loading");
    else if (query.isError) setLifecycle("error");
  }, [query.isLoading, query.isError, setLifecycle]);

  return {
    wafer: waferId && wafer?.wafer_id === waferId ? wafer : query.data ? mapWaferDetail(query.data) : null,
    lifecycle,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}
