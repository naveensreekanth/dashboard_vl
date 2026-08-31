"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * Mount heavy KPI visuals only after the card enters the viewport.
 * Keeps the optimization grid light on first paint / open.
 */
export function DeferredMount({
  children,
  className,
  minHeight,
  rootMargin = "120px",
}: {
  children: ReactNode;
  className?: string;
  minHeight?: number | string;
  rootMargin?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const schedule = () => {
      timer = setTimeout(() => {
        if (!cancelled) setReady(true);
      }, 40);
    };

    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry?.isIntersecting) return;
        io.disconnect();
        schedule();
      },
      { rootMargin, threshold: 0.08 },
    );
    io.observe(el);

    return () => {
      cancelled = true;
      io.disconnect();
      if (timer != null) clearTimeout(timer);
    };
  }, [rootMargin]);

  return (
    <div
      ref={ref}
      className={className}
      style={minHeight != null ? { minHeight } : undefined}
    >
      {ready ? (
        children
      ) : (
        <div
          className="vl-tto-skeleton h-full min-h-[120px] w-full rounded-[5px]"
          aria-hidden
        />
      )}
    </div>
  );
}
