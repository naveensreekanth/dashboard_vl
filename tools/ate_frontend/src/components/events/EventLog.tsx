import type { EventLogItem } from "@/types/api";
import { formatTime } from "@/lib/utils";
import { EmptyState } from "@/components/common/EmptyState";

export function EventLog({ events }: { events: EventLogItem[] }) {
  return (
    <div className="rounded border border-[var(--line)] bg-[var(--panel)] p-5">
      <h2 className="mb-3.5 flex justify-between text-[12px] font-semibold uppercase tracking-[0.1em] text-[var(--muted)]">
        <span>Test Floor Event Log</span>
        <span className="font-normal normal-case tracking-normal text-[var(--muted-2)]">
          {events.length} events
        </span>
      </h2>
      {events.length === 0 ? (
        <EmptyState message="No floor events yet." />
      ) : (
        <div className="flex max-h-[230px] flex-col overflow-y-auto">
          {events.map((ev) => (
            <div
              key={ev.event_id}
              className="grid grid-cols-[78px_100px_1fr] items-start gap-3 border-t border-[var(--line)] py-2 text-[11.5px] first:border-t-0"
            >
              <div className="font-mono text-[var(--muted-2)]">{formatTime(ev.timestamp)}</div>
              <div>
                <span
                  className={`inline-block w-fit rounded-[3px] px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.03em] ${
                    ev.tag === "pass"
                      ? "bg-[var(--green-dim)] text-[var(--green)]"
                      : ev.tag === "warn"
                        ? "bg-[var(--amber-dim)] text-[var(--amber)]"
                        : "bg-[var(--cyan-dim)] text-[var(--cyan)]"
                  }`}
                >
                  {ev.tag}
                </span>
              </div>
              <div className="text-[var(--text)]">{ev.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
