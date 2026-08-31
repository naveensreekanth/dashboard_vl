import { create } from "zustand";
import type { DashboardSummary, DieOut } from "@/types/api";

interface DashboardState {
  summary: DashboardSummary | null;
  dies: DieOut[];
  setSummary: (summary: DashboardSummary) => void;
  setDies: (dies: DieOut[]) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  summary: null,
  dies: [],
  setSummary: (summary) => set({ summary }),
  setDies: (dies) => set({ dies }),
}));
