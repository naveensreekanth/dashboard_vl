"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { fetchLatestShmoo } from "@/services/api";
import { useShmooStore } from "@/stores/shmooStore";

/**
 * Keep SHMOO KPI card plots/results live from the latest server upload.
 */
export function useLatestShmoo(enabled = true) {
  const setSession = useShmooStore((s) => s.setSession);

  const query = useQuery({
    queryKey: ["shmoo", "latest"],
    queryFn: fetchLatestShmoo,
    enabled,
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
  });

  useEffect(() => {
    const data = query.data;
    if (!data?.session_id || !data.results || !data.meta || !data.plot_url) return;

    setSession({
      sessionId: data.session_id,
      filename: data.filename,
      meta: data.meta,
      results: data.results,
      plotUrl: data.plot_url,
      plotUrls: {
        character: data.plot_urls?.character ?? data.plot_url,
        yield: data.plot_urls?.yield ?? null,
        debug: data.plot_urls?.debug ?? null,
      },
    });
  }, [query.data, setSession]);

  return query;
}
