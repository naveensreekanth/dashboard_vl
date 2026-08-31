/** Presentation metrics matching the Test Time Optimization agent UI. */

export type TestTimeCompareMetric = {
  id: string;
  label: string;
  value: number;
  unit: string;
  /** Relative bar fill 0–1 (baseline = 1). */
  bar: number;
  tone: "baseline" | "optimized";
};

export const TEST_TIME_COMPARE_METRICS: TestTimeCompareMetric[] = [
  {
    id: "vram_without",
    label: "without verilumen agent vector RAM",
    value: 14.3,
    unit: "MB",
    bar: 1,
    tone: "baseline",
  },
  {
    id: "vram_with",
    label: "with verilumen agent vector RAM",
    value: 8.63,
    unit: "MB",
    bar: 8.63 / 14.3,
    tone: "optimized",
  },
  {
    id: "tt_without",
    label: "without verilumen agent test time",
    value: 24.84,
    unit: "ms",
    bar: 1,
    tone: "baseline",
  },
  {
    id: "tt_with",
    label: "with verilumen agent test time",
    value: 14.99,
    unit: "ms",
    bar: 14.99 / 24.84,
    tone: "optimized",
  },
];

export type LiveMemoryPoint = {
  /** Continuous x index across both cycles (for plotting). */
  i: number;
  /** Tick label shown on axis (resets per cycle). */
  tick: number;
  without: number;
  withAgent: number;
};

const PHASE1 = 950;
const PHASE2 = 1000;
const PEAK_WITHOUT = 14.3;
const PEAK_WITH = 8.63;

/**
 * Downsampled live vector-memory series:
 * cycle 1 — without climbs, with stays ~0; reset; cycle 2 — both climb (with shallower).
 */
export function buildLiveVectorMemorySeries(step = 25): LiveMemoryPoint[] {
  const points: LiveMemoryPoint[] = [];
  let i = 0;

  for (let t = 1; t <= PHASE1; t += step) {
    const frac = t / PHASE1;
    points.push({
      i: i++,
      tick: t,
      without: PEAK_WITHOUT * frac,
      withAgent: 0,
    });
  }

  // Sharp reset “flash” at cycle boundary
  points.push({ i: i++, tick: PHASE1, without: 0, withAgent: 0 });
  points.push({ i: i++, tick: 1, without: 0, withAgent: 0 });

  for (let t = 1; t <= PHASE2; t += step) {
    const frac = t / PHASE2;
    // Mild noise on green to match agent trace look
    const wobble = Math.sin(t / 37) * 0.12 + Math.sin(t / 11) * 0.06;
    points.push({
      i: i++,
      tick: t,
      without: PEAK_WITHOUT * frac,
      withAgent: Math.max(0, PEAK_WITH * frac + wobble * frac),
    });
  }

  return points;
}

export const TEST_TIME_LIVE_MEMORY = buildLiveVectorMemorySeries(25);
/** Card embed — fewer points so the KPI grid stays responsive. */
export const TEST_TIME_LIVE_MEMORY_COMPACT = buildLiveVectorMemorySeries(80);
