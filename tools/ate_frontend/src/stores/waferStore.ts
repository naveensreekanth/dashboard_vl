import { create } from "zustand";
import { applyDieTelemetryEvent } from "@/lib/waferMappers";
import type { Die, Wafer, WaferLifecycleStatus, WaferTelemetryEvent } from "@/types/wafer";
import { DIE_EVENT_TYPES } from "@/types/wafer";

interface WaferClientState {
  waferId: string | null;
  wafer: Wafer | null;
  /** die_id → Die — live overlay patched by WebSocket (affected die only) */
  diesById: Record<string, Die>;
  selectedDieId: string | null;
  hoveredDieId: string | null;
  lifecycle: WaferLifecycleStatus;
  lastSequence: number;
  setWaferId: (waferId: string | null) => void;
  hydrateWafer: (wafer: Wafer) => void;
  hydrateDies: (dies: Die[]) => void;
  setLifecycle: (lifecycle: WaferLifecycleStatus) => void;
  setHoveredDieId: (dieId: string | null) => void;
  selectDie: (dieId: string | null) => void;
  applyTelemetryEvent: (event: WaferTelemetryEvent) => void;
  applyYieldUpdate: (event: WaferTelemetryEvent) => void;
}

export const useWaferStore = create<WaferClientState>((set, get) => ({
  waferId: null,
  wafer: null,
  diesById: {},
  selectedDieId: null,
  hoveredDieId: null,
  lifecycle: "loading",
  lastSequence: 0,

  setWaferId: (waferId) => set({ waferId }),

  hydrateWafer: (wafer) =>
    set((state) => ({
      wafer,
      waferId: wafer.wafer_id,
      lifecycle:
        state.lifecycle === "offline" || state.lifecycle === "error"
          ? state.lifecycle
          : wafer.status === "completed"
            ? "completed"
            : "live",
    })),

  hydrateDies: (dies) => {
    const diesById: Record<string, Die> = {};
    for (const d of dies) diesById[d.die_id] = d;
    set((state) => ({
      diesById,
      lifecycle:
        dies.length === 0
          ? "empty"
          : state.lifecycle === "offline" || state.lifecycle === "error"
            ? state.lifecycle
            : state.wafer?.status === "completed"
              ? "completed"
              : "live",
    }));
  },

  setLifecycle: (lifecycle) => set({ lifecycle }),

  setHoveredDieId: (hoveredDieId) => set({ hoveredDieId }),

  selectDie: (selectedDieId) => set({ selectedDieId }),

  applyTelemetryEvent: (event) => {
    if (!DIE_EVENT_TYPES.includes(event.event_type as (typeof DIE_EVENT_TYPES)[number])) {
      return;
    }
    const { waferId, diesById, lastSequence } = get();
    if (!event.wafer_id || (waferId && event.wafer_id !== waferId)) return;
    if (event.sequence_number > 0 && event.sequence_number <= lastSequence) return;

    const keyGuess =
      event.die_id && diesById[event.die_id]
        ? event.die_id
        : Object.keys(diesById).find((id) => {
            const d = diesById[id];
            const x = event.payload.x;
            const y = event.payload.y;
            return d && x != null && y != null && d.x === x && d.y === y;
          });

    const prev = keyGuess ? diesById[keyGuess] : undefined;
    const next = applyDieTelemetryEvent(prev, event);
    if (!next) return;

    // Patch only the affected die — wafer aggregates come from backend yield_updated / REST
    set({
      diesById: { ...diesById, [next.die_id]: next },
      lastSequence: Math.max(lastSequence, event.sequence_number),
      lifecycle: "live",
    });
  },

  applyYieldUpdate: (event) => {
    const { wafer, waferId } = get();
    if (!wafer || !event.wafer_id || (waferId && event.wafer_id !== waferId)) return;
    const p = event.payload;
    const testedFromBins =
      typeof p.pass === "number"
        ? (p.pass ?? 0) + (p.fail ?? 0) + (p.retest ?? 0) + (p.reclass ?? 0)
        : null;

    set({
      wafer: {
        ...wafer,
        yield_pct: typeof p.yield_pct === "number" ? p.yield_pct : wafer.yield_pct,
        pass_count: typeof p.pass === "number" ? p.pass : wafer.pass_count,
        fail_count: typeof p.fail === "number" ? p.fail : wafer.fail_count,
        retest_count: typeof p.retest === "number" ? p.retest : wafer.retest_count,
        reclass_count: typeof p.reclass === "number" ? p.reclass : wafer.reclass_count,
        total_dies: typeof p.total === "number" ? p.total : wafer.total_dies,
        tested_dies:
          typeof p.tested_dies === "number"
            ? p.tested_dies
            : testedFromBins != null
              ? testedFromBins
              : wafer.tested_dies,
        status:
          event.event_type === "lot_completed" || event.event_type === "wafer_completed"
            ? "completed"
            : wafer.status,
      },
      lastSequence: Math.max(get().lastSequence, event.sequence_number),
      lifecycle:
        event.event_type === "lot_completed" || wafer.status === "completed" ? "completed" : "live",
    });
  },
}));
