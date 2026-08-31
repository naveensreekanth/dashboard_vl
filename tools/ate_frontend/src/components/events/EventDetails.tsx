"use client";

import { DetailPopup } from "@/components/common/DetailPopup";
import { formatTime } from "@/lib/utils";
import type { TestEvent } from "@/types/events";
import { SEVERITY_STYLES } from "@/types/events";

interface Props {
  event: TestEvent | null;
  onClose: () => void;
  onAcknowledge: (eventId: string) => void;
  acknowledging?: boolean;
}

export function EventDetails({ event, onClose, onAcknowledge, acknowledging }: Props) {
  if (!event) return null;

  const style = SEVERITY_STYLES[event.severity];

  return (
    <DetailPopup
      eyebrow="Test Floor Event"
      title={event.event_type.replace(/_/g, " ")}
      onClose={onClose}
      wide
    >
      <div className="mb-1 font-mono text-[11px] text-[#9ec9ef]">{event.event_id}</div>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span
          className="rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.06em]"
          style={{ color: style.fg, background: style.bg }}
        >
          {event.severity}
        </span>
        <span className="rounded border border-[rgba(107,193,242,0.35)] bg-[rgba(107,193,242,0.12)] px-2 py-0.5 font-mono text-[10px] text-[#c9e6ff]">
          {event.event_type}
        </span>
        <span className="font-mono text-[11px] text-[#b7d4f0]">
          seq {event.sequence_number}
        </span>
      </div>

      <p className="mb-4 text-[13px] leading-relaxed text-[#f2f7fc]">{event.message}</p>

      <div className="mb-4 grid grid-cols-2 gap-2.5 sm:grid-cols-3">
        <Field label="Timestamp" value={formatTime(event.timestamp)} />
        <Field label="Source" value={event.source} />
        <Field label="Tester" value={event.tester_id} />
        <Field label="Site" value={event.site_id} />
        <Field label="Lot" value={event.lot_id} />
        <Field label="Wafer" value={event.wafer_id} />
        <Field label="Die" value={event.die_id} />
        <Field
          label="Ack"
          value={
            event.acknowledged
              ? `${event.acknowledged_by ?? "yes"}${
                  event.acknowledged_at ? ` · ${formatTime(event.acknowledged_at)}` : ""
                }`
              : "open"
          }
          accent={event.acknowledged ? "var(--green)" : "var(--amber)"}
        />
      </div>

      <div className="vl-popup-label mb-2">Metadata</div>
      <div className="vl-popup-tile mb-4 p-3">
        <pre className="whitespace-pre-wrap break-all font-mono text-[11px] leading-relaxed text-[#e4f0fb]">
          {JSON.stringify(event.metadata ?? {}, null, 2)}
        </pre>
      </div>

      <button
        type="button"
        disabled={event.acknowledged || acknowledging}
        onClick={() => onAcknowledge(event.event_id)}
        className="rounded-[6px] border border-[rgba(107,193,242,0.55)] bg-[rgba(107,193,242,0.14)] px-3 py-2 text-[12px] font-semibold text-[#c9e6ff] transition-colors hover:border-[var(--cyan)] hover:text-white disabled:opacity-40"
      >
        {event.acknowledged ? "Acknowledged" : acknowledging ? "Acknowledging…" : "Acknowledge"}
      </button>
    </DetailPopup>
  );
}

function Field({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | null | undefined;
  accent?: string;
}) {
  return (
    <div className="vl-popup-tile px-3 py-2.5">
      <dt className="vl-popup-tile-label">{label}</dt>
      <dd
        className="vl-popup-tile-value mt-1 font-mono text-[12px] font-semibold"
        style={accent ? { color: accent } : undefined}
      >
        {value ?? "—"}
      </dd>
    </div>
  );
}
