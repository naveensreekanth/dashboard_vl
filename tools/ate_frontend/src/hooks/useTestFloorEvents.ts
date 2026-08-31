"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import {
  acknowledgeTestEvent,
  fetchEventFilterOptions,
  fetchTestEvents,
} from "@/services/api";
import { useTestEventStore } from "@/stores/testEventStore";
import type { EventFiltersState, TestEvent } from "@/types/events";

function matchesFilters(event: TestEvent, filters: EventFiltersState): boolean {
  if (filters.tester_id && event.tester_id !== filters.tester_id) return false;
  if (filters.site_id && event.site_id !== filters.site_id) return false;
  if (filters.lot_id && event.lot_id !== filters.lot_id) return false;
  if (filters.wafer_id && event.wafer_id !== filters.wafer_id) return false;
  if (filters.severity.length && !filters.severity.includes(event.severity)) return false;
  if (filters.acknowledged === "true" && !event.acknowledged) return false;
  if (filters.acknowledged === "false" && event.acknowledged) return false;
  if (filters.since && new Date(event.timestamp) < new Date(filters.since)) return false;
  if (filters.until && new Date(event.timestamp) > new Date(filters.until)) return false;
  if (filters.q) {
    const q = filters.q.toLowerCase();
    const hay = [
      event.message,
      event.event_type,
      event.source,
      event.tester_id,
      event.site_id,
      event.lot_id,
      event.wafer_id,
      event.die_id,
      event.severity,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

function mergeByEventId(historical: TestEvent[], live: TestEvent[]): TestEvent[] {
  const map = new Map<string, TestEvent>();
  for (const e of historical) map.set(e.event_id, e);
  for (const e of live) {
    const prev = map.get(e.event_id);
    // Prefer acknowledged / richer historical row when IDs collide
    if (!prev) {
      map.set(e.event_id, e);
    } else {
      map.set(e.event_id, {
        ...e,
        ...prev,
        acknowledged: prev.acknowledged || e.acknowledged,
        message: prev.message || e.message,
        metadata: { ...e.metadata, ...prev.metadata },
      });
    }
  }
  return [...map.values()].sort((a, b) => {
    if (b.sequence_number !== a.sequence_number) {
      return b.sequence_number - a.sequence_number;
    }
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });
}

export function useTestFloorEvents(filters: EventFiltersState) {
  const queryClient = useQueryClient();
  const liveById = useTestEventStore((s) => s.liveById);
  const markAcknowledged = useTestEventStore((s) => s.markAcknowledged);

  const historyQuery = useQuery({
    queryKey: ["test-events", filters],
    queryFn: () => fetchTestEvents({ ...filters, limit: 200 }),
    refetchInterval: 20_000,
  });

  const optionsQuery = useQuery({
    queryKey: ["test-events", "filters"],
    queryFn: fetchEventFilterOptions,
    staleTime: 60_000,
  });

  const liveList = useMemo(() => Object.values(liveById), [liveById]);

  const events = useMemo(() => {
    const historical = historyQuery.data?.items ?? [];
    const liveFiltered = liveList.filter((e) => matchesFilters(e, filters));
    return mergeByEventId(historical, liveFiltered);
  }, [historyQuery.data?.items, liveList, filters]);

  const ackMutation = useMutation({
    mutationFn: (eventId: string) =>
      acknowledgeTestEvent(eventId, { actor: "engineer" }),
    onSuccess: (updated) => {
      markAcknowledged(updated.event_id, updated);
      void queryClient.invalidateQueries({ queryKey: ["test-events"] });
    },
  });

  return {
    events,
    total: historyQuery.data?.total ?? events.length,
    unacknowledged:
      historyQuery.data?.unacknowledged ??
      events.filter((e) => !e.acknowledged).length,
    filterOptions: optionsQuery.data,
    isLoading: historyQuery.isLoading,
    isError: historyQuery.isError,
    refetch: historyQuery.refetch,
    acknowledge: (eventId: string) => ackMutation.mutate(eventId),
    acknowledging: ackMutation.isPending,
  };
}
