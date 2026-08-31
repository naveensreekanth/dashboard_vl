export type ShmooPointKind = "pass" | "freq_margin" | "timing";

export type ShmooDemoPoint = {
  vdd: number;
  freq: number;
  kind: ShmooPointKind;
};

export const SHMOO_DEMO_VDD_RANGE: [number, number] = [0.79, 1.0];
export const SHMOO_DEMO_FREQ_RANGE: [number, number] = [0.92, 2.06];

/** Recommended OP fixed at plot center (mid VDD / mid Frequency). */
export const SHMOO_DEMO_OP = {
  vdd: (SHMOO_DEMO_VDD_RANGE[0] + SHMOO_DEMO_VDD_RANGE[1]) / 2,
  freq: (SHMOO_DEMO_FREQ_RANGE[0] + SHMOO_DEMO_FREQ_RANGE[1]) / 2,
};

/**
 * Population boundary: freq ≈ slope * vdd + intercept
 * Corner-to-corner through Recommended OP (plot center):
 * (0.79, 0.92) → (1.00, 2.06)
 */
const _v0 = SHMOO_DEMO_VDD_RANGE[0];
const _v1 = SHMOO_DEMO_VDD_RANGE[1];
const _f0 = SHMOO_DEMO_FREQ_RANGE[0];
const _f1 = SHMOO_DEMO_FREQ_RANGE[1];
const _slope = (_f1 - _f0) / (_v1 - _v0);
export const SHMOO_DEMO_BOUNDARY = {
  slope: _slope,
  intercept: _f0 - _slope * _v0,
};

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x5a17e2);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Deterministic ensemble SHMOO — sharp linear cliff matching reference look.
 * Solid vertical columns; orange only on the pass/fail transition cell.
 */
export function buildShmooDemoScatter(): ShmooDemoPoint[] {
  const rand = mulberry32(0x5a17e2);
  const [vMin, vMax] = SHMOO_DEMO_VDD_RANGE;
  const [fMin, fMax] = SHMOO_DEMO_FREQ_RANGE;
  const cols = 20;
  const rows = 26;
  const cellH = (fMax - fMin) / rows;
  const points: ShmooDemoPoint[] = [];

  for (let c = 0; c < cols; c++) {
    const vdd = vMin + ((vMax - vMin) * (c + 0.5)) / cols;
    // Tiny per-column edge wobble only — keeps a clean cliff
    const edge =
      SHMOO_DEMO_BOUNDARY.slope * vdd +
      SHMOO_DEMO_BOUNDARY.intercept +
      (rand() - 0.5) * 0.018;

    for (let r = 0; r < rows; r++) {
      const freq = fMin + ((fMax - fMin) * (r + 0.5)) / rows;
      const delta = freq - edge;
      let kind: ShmooPointKind;
      if (delta < -cellH * 0.35) {
        kind = "pass";
      } else if (delta < cellH * 0.65) {
        // Single transition band — occasional timing/defect
        kind = rand() < 0.42 ? "timing" : "freq_margin";
      } else {
        kind = "freq_margin";
      }
      points.push({ vdd, freq, kind });
    }
  }
  return points;
}

export const SHMOO_DEMO_POINTS = buildShmooDemoScatter();
