import { create } from "zustand";
import type { Kpi } from "@/types/kpi";

interface KpiClientState {
  kpisById: Record<string, Kpi>;
  selectedKpiId: string | null;
  hydrateKpis: (kpis: Kpi[]) => void;
  patchKpi: (kpi: Kpi) => void;
  selectKpi: (kpiId: string | null) => void;
}

export const useKpiStore = create<KpiClientState>((set) => ({
  kpisById: {},
  selectedKpiId: null,
  hydrateKpis: (kpis) => {
    const kpisById: Record<string, Kpi> = {};
    for (const k of kpis) kpisById[k.id] = k;
    set({ kpisById });
  },
  patchKpi: (kpi) =>
    set((state) => ({
      kpisById: { ...state.kpisById, [kpi.id]: kpi },
    })),
  selectKpi: (selectedKpiId) => set({ selectedKpiId }),
}));
