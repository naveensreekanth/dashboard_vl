"use client";

import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { LoadingState } from "@/components/common/LoadingState";
import { EventDetails } from "@/components/events/EventDetails";
import { EventFilters } from "@/components/events/EventFilters";
import { useTestFloorEvents } from "@/hooks/useTestFloorEvents";
import { formatTime } from "@/lib/utils";
import { useConnectionStore } from "@/stores/connectionStore";
import { useOpsStore } from "@/stores/opsStore";
import type { EventFiltersState, TestEvent } from "@/types/events";
import { EMPTY_EVENT_FILTERS, SEVERITY_STYLES } from "@/types/events";

export function TestFloorEventLog() {
  const [filters, setFilters] = useState<EventFiltersState>(EMPTY_EVENT_FILTERS);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const status = useConnectionStore((s) => s.status);
  const lotId = useOpsStore((s) => s.lotId);
  const waferId = useOpsStore((s) => s.waferId);
  const testerId = useOpsStore((s) => s.testerId);
  const siteId = useOpsStore((s) => s.siteId);
  const since = useOpsStore((s) => s.since);
  const until = useOpsStore((s) => s.until);

  useEffect(() => {
    setFilters((prev) => ({
      ...prev,
      lot_id: lotId,
      wafer_id: waferId,
      tester_id: testerId,
      site_id: siteId,
      since,
      until,
    }));
  }, [lotId, waferId, testerId, siteId, since, until]);

  const {
    events,
    total,
    unacknowledged,
    filterOptions,
    isLoading,
    isError,
    refetch,
    acknowledge,
    acknowledging,
  } = useTestFloorEvents(filters);

  const selected = useMemo(
    () => events.find((e) => e.event_id === selectedId) ?? null,
    [events, selectedId],
  );

  return (
    <section className="vl-surface p-5">
      <header className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="vl-section-title m-0">Test Floor Event Log</h2>
          <p className="mt-1.5 text-[11px] text-[#8fa6c0]">
            Authoritative backend events · WebSocket live insert · React Query history
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5 font-mono text-[11px] text-[#9eb6d0]">
          <span className="vl-chip py-1">
            <span className="text-white">{events.length}</span> shown
          </span>
          <span className="vl-chip py-1">
            <span className="text-white">{total}</span> total
          </span>
          <span className="vl-chip py-1">
            <span className="text-[var(--amber)]">{unacknowledged}</span> open
          </span>
          <span
            className={
              status === "LIVE"
                ? "vl-live-badge rounded px-2 py-1"
                : `rounded px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] ${
                    status === "OFFLINE"
                      ? "bg-[var(--red-dim)] text-[var(--red)]"
                      : "bg-[var(--amber-dim)] text-[var(--amber)]"
                  }`
            }
          >
            {status === "LIVE" ? (
              <>
                <span className="vl-live-dot" />
                LIVE
              </>
            ) : (
              status
            )}
          </span>
        </div>
      </header>

      <EventFilters filters={filters} options={filterOptions} onChange={setFilters} />

      <div className="mt-3 min-h-[260px]">
        {isLoading && events.length === 0 ? (
          <LoadingState />
        ) : isError && events.length === 0 ? (
          <ErrorState
            message="Unable to load historical floor events from the API."
            onRetry={() => void refetch()}
          />
        ) : events.length === 0 ? (
          <EmptyState message="No events match the current filters." />
        ) : (
          <div className="max-h-[420px] overflow-y-auto rounded-[8px] border border-[rgba(107,193,242,0.22)]">
            <div className="sticky top-0 grid grid-cols-[72px_72px_88px_1fr_64px] gap-2 border-b border-[rgba(107,193,242,0.22)] bg-[#101826] px-2.5 py-2 text-[9.5px] font-semibold uppercase tracking-[0.1em] text-[#9ec9ef]">
              <span>Time</span>
              <span>Sev</span>
              <span>Type</span>
              <span>Message</span>
              <span>Ack</span>
            </div>
            {events.map((ev) => (
              <EventRow
                key={ev.event_id}
                event={ev}
                selected={ev.event_id === selectedId}
                onSelect={() => setSelectedId(ev.event_id)}
              />
            ))}
          </div>
        )}
        <p className="mt-2 text-[11px] text-[var(--muted-2)]">
          Click a row to open the event drill-down popup.
        </p>
      </div>

      <EventDetails
        event={selected}
        onClose={() => setSelectedId(null)}
        onAcknowledge={acknowledge}
        acknowledging={acknowledging}
      />
    </section>
  );
}

function EventRow({
  event,
  selected,
  onSelect,
}: {
  event: TestEvent;
  selected: boolean;
  onSelect: () => void;
}) {
  const style = SEVERITY_STYLES[event.severity];
  const isCritical = event.severity === "CRITICAL" || event.severity === "ERROR";
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`vl-enter-row grid w-full grid-cols-[72px_72px_88px_1fr_64px] items-start gap-2 border-t border-[rgba(107,193,242,0.12)] px-2.5 py-2.5 text-left text-[11.5px] first:border-t-0 transition-[background-color,box-shadow] duration-200 ${
        selected ? "bg-[rgba(107,193,242,0.14)]" : "hover:bg-[rgba(107,193,242,0.06)]"
      } ${event.acknowledged ? "opacity-70" : ""} ${
        isCritical && !event.acknowledged
          ? "shadow-[inset_2px_0_0_0_rgba(240,102,122,0.55)]"
          : ""
      }`}
    >
      <span className="font-mono text-[10.5px] text-[var(--muted-2)]">
        {formatTime(event.timestamp)}
      </span>
      <span>
        <span
          className="inline-block rounded px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.03em]"
          style={{ color: style.fg, background: style.bg }}
        >
          {event.severity}
        </span>
      </span>
      <span className="truncate font-mono text-[10px] text-[var(--muted)]">{event.event_type}</span>
      <span className="truncate text-[var(--text)]">{event.message}</span>
      <span className="font-mono text-[10px] text-[var(--muted-2)]">
        {event.acknowledged ? "ACK" : "OPEN"}
      </span>
    </button>
  );
}
