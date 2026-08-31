"use client";

import { useEffect, useRef, useState } from "react";
import { formatNumber } from "@/lib/utils";

function easeOutCubic(t: number): number {
  return 1 - (1 - t) ** 3;
}

/**
 * Smooth count-up / interpolate for instrumentation-style KPI numbers.
 * Respects prefers-reduced-motion.
 */
export function AnimatedNumber({
  value,
  digits = 1,
  className,
  durationMs = 750,
}: {
  value: number;
  digits?: number;
  className?: string;
  durationMs?: number;
}) {
  const [display, setDisplay] = useState(() =>
    Number.isFinite(value) ? value * 0.82 : value,
  );
  const [flash, setFlash] = useState(false);
  const fromRef = useRef(Number.isFinite(value) ? value * 0.82 : value);
  const mountedRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (!Number.isFinite(value)) {
      setDisplay(value);
      return;
    }

    if (reduced) {
      setDisplay(value);
      fromRef.current = value;
      mountedRef.current = true;
      return;
    }

    const from = mountedRef.current ? fromRef.current : value * 0.82;
    const to = value;
    mountedRef.current = true;
    const start = performance.now();

    if (from !== to && fromRef.current !== value * 0.82) {
      setFlash(true);
      if (flashTimer.current) clearTimeout(flashTimer.current);
      flashTimer.current = setTimeout(() => setFlash(false), 340);
    }

    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      setDisplay(from + (to - from) * easeOutCubic(t));
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = to;
        setDisplay(to);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      if (flashTimer.current) clearTimeout(flashTimer.current);
    };
  }, [value, durationMs]);

  if (!Number.isFinite(value)) {
    return <span className={className}>—</span>;
  }

  return (
    <span className={`${className ?? ""} ${flash ? "vl-num-flash" : ""}`.trim()}>
      {formatNumber(display, digits)}
    </span>
  );
}
