import type { TelemetryEvent } from "@/types/api";
import type { TestEvent } from "@/types/events";
import { FLOOR_EVENT_TYPES } from "@/types/events";

/**
 * Prefer authoritative `test_event` from the backend WS envelope.
 * Never invent floor events client-side — only accept server-authored snapshots
 * or skip live insertion (historical RQ refetch will catch up).
 */
export function testEventFromWsEnvelope(msg: {
  test_event?: TestEvent | null;
  event?: TelemetryEvent | null;
}): TestEvent | null {
  const te = msg.test_event;
  if (te?.event_id && te.message != null && te.severity) {
    return {
      ...te,
      metadata: te.metadata ?? {},
      acknowledged: Boolean(te.acknowledged),
      sequence_number: te.sequence_number ?? 0,
    };
  }

  // No client-side event synthesis — die noise / missing snapshot is ignored
  const event = msg.event;
  if (!event || !FLOOR_EVENT_TYPES.has(event.event_type)) return null;
  return null;
}
