import { create } from "zustand";
import type { OpsConnectionStatus } from "@/types/events";

interface ConnectionState {
  status: OpsConnectionStatus;
  lastError: string | null;
  lastMessageAt: number | null;
  lastSequence: number | null;
  setStatus: (status: OpsConnectionStatus) => void;
  setError: (error: string | null) => void;
  touchMessage: (sequence?: number | null) => void;
  secondsSinceTelemetry: () => number | null;
  forceOffline: () => void;
}

export const useConnectionStore = create<ConnectionState>((set, get) => ({
  status: "OFFLINE",
  lastError: null,
  lastMessageAt: null,
  lastSequence: null,
  setStatus: (status) => set({ status }),
  setError: (lastError) => set({ lastError }),
  touchMessage: (sequence) =>
    set((s) => ({
      lastMessageAt: Date.now(),
      lastSequence:
        typeof sequence === "number"
          ? Math.max(s.lastSequence ?? 0, sequence)
          : s.lastSequence,
    })),
  secondsSinceTelemetry: () => {
    const t = get().lastMessageAt;
    if (t == null) return null;
    return Math.max(0, Math.floor((Date.now() - t) / 1000));
  },
  forceOffline: () => set({ status: "OFFLINE", lastError: "Stream paused or disconnected" }),
}));
