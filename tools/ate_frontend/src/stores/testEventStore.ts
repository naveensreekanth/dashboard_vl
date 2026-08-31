import { create } from "zustand";
import type { TestEvent } from "@/types/events";

interface TestEventState {
  liveById: Record<string, TestEvent>;
  liveOrder: string[];
  upsertLive: (event: TestEvent) => void;
  markAcknowledged: (eventId: string, patch: Partial<TestEvent>) => void;
  clearLive: () => void;
}

const MAX_LIVE = 500;

export const useTestEventStore = create<TestEventState>((set) => ({
  liveById: {},
  liveOrder: [],
  upsertLive: (event) =>
    set((state) => {
      const exists = Boolean(state.liveById[event.event_id]);
      const liveById = { ...state.liveById, [event.event_id]: event };
      let liveOrder = exists
        ? state.liveOrder
        : [event.event_id, ...state.liveOrder];
      if (liveOrder.length > MAX_LIVE) {
        const drop = liveOrder.slice(MAX_LIVE);
        liveOrder = liveOrder.slice(0, MAX_LIVE);
        for (const id of drop) delete liveById[id];
      }
      return { liveById, liveOrder };
    }),
  markAcknowledged: (eventId, patch) =>
    set((state) => {
      const cur = state.liveById[eventId];
      if (!cur) return state;
      return {
        liveById: { ...state.liveById, [eventId]: { ...cur, ...patch } },
      };
    }),
  clearLive: () => set({ liveById: {}, liveOrder: [] }),
}));
