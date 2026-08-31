import { create } from "zustand";
import type { ShmooMeta, ShmooResults } from "@/types/shmoo";

export type ShmooPlotUrls = {
  character: string | null;
  yield: string | null;
  debug: string | null;
};

interface ShmooState {
  sessionId: string | null;
  filename: string | null;
  meta: ShmooMeta | null;
  results: ShmooResults | null;
  plotUrl: string | null;
  plotUrls: ShmooPlotUrls;
  setSession: (payload: {
    sessionId: string;
    filename?: string;
    meta: ShmooMeta;
    results: ShmooResults;
    plotUrl: string;
    plotUrls?: Partial<ShmooPlotUrls>;
  }) => void;
  clear: () => void;
}

const emptyPlots: ShmooPlotUrls = {
  character: null,
  yield: null,
  debug: null,
};

export const useShmooStore = create<ShmooState>((set) => ({
  sessionId: null,
  filename: null,
  meta: null,
  results: null,
  plotUrl: null,
  plotUrls: { ...emptyPlots },
  setSession: ({ sessionId, filename, meta, results, plotUrl, plotUrls }) =>
    set({
      sessionId,
      filename: filename ?? null,
      meta,
      results,
      plotUrl,
      plotUrls: {
        character: plotUrls?.character ?? plotUrl ?? null,
        yield: plotUrls?.yield ?? null,
        debug: plotUrls?.debug ?? null,
      },
    }),
  clear: () =>
    set({
      sessionId: null,
      filename: null,
      meta: null,
      results: null,
      plotUrl: null,
      plotUrls: { ...emptyPlots },
    }),
}));
