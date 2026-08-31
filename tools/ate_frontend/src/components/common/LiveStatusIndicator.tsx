"use client";

import { useEffect, useState } from "react";
import { useConnectionStore } from "@/stores/connectionStore";
import type { OpsConnectionStatus } from "@/types/events";

const STYLE: Record<OpsConnectionStatus, string> = {
  LIVE: "var(--green)",
  RECONNECTING: "var(--amber)",
  DEGRADED: "var(--amber)",
  STALE: "var(--amber)",
  OFFLINE: "var(--red)",
};

export function LiveStatusIndicator() {
  const status = useConnectionStore((s) => s.status);
  const lastMessageAt = useConnectionStore((s) => s.lastMessageAt);
  const [ago, setAgo] = useState<number | null>(null);
  const color = STYLE[status];

  useEffect(() => {
    const tick = () => {
      if (lastMessageAt == null) {
        setAgo(null);
        return;
      }
      setAgo(Math.max(0, Math.floor((Date.now() - lastMessageAt) / 1000)));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [lastMessageAt]);

  return (
    <div className="flex flex-col gap-1 text-[11px] uppercase tracking-[0.1em] text-[var(--muted-2)]">
      <div className="flex items-center gap-2">
        <span
          className={`inline-block h-2 w-2 rounded-full ${status === "LIVE" ? "vl-live-dot" : ""}`}
          style={{
            background: color,
            boxShadow: status === "LIVE" ? undefined : `0 0 8px ${color}`,
          }}
        />
        <span className="font-semibold" style={{ color }}>
          {status}
        </span>
      </div>
      <div className="normal-case tracking-normal text-[10px] text-[var(--muted)]">
        Last telemetry received:{" "}
        <span className="font-mono text-[var(--text)]">
          {ago == null ? "—" : `${ago}s ago`}
        </span>
      </div>
    </div>
  );
}
