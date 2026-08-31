"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { testEventFromWsEnvelope } from "@/lib/mapTelemetryToTestEvent";
import { useAuthStore } from "@/stores/authStore";
import { useConnectionStore } from "@/stores/connectionStore";
import { useOpsStore } from "@/stores/opsStore";
import { useTestEventStore } from "@/stores/testEventStore";
import { useWaferStore } from "@/stores/waferStore";
import type { TelemetryEvent } from "@/types/api";
import { KPI_LIVE_EVENTS } from "@/types/kpi";
import type { WaferTelemetryEvent } from "@/types/wafer";
import { DIE_EVENT_TYPES } from "@/types/wafer";

const DEFAULT_PROD_WS = "wss://wafer-yield-api.onrender.com/ws/test-floor";

/**
 * Always return a real ws/wss URL for the test-floor stream.
 * Guards against mis-set Vercel env (REST /api URL pasted into NEXT_PUBLIC_WS_URL).
 */
function resolveWsBase(): string {
  const raw = (process.env.NEXT_PUBLIC_WS_URL || "").trim();

  if (raw) {
    // Common misconfig: REST base used as WS → causes "WebSocket error"
    const looksLikeRestApi =
      raw.includes("/api") && !raw.includes("/ws/");
    if (looksLikeRestApi) {
      return DEFAULT_PROD_WS;
    }
    if (raw.startsWith("https://")) return `wss://${raw.slice("https://".length)}`;
    if (raw.startsWith("http://")) return `ws://${raw.slice("http://".length)}`;
    if (raw.startsWith("wss://") || raw.startsWith("ws://")) return raw;
  }

  if (typeof window === "undefined") return "ws://127.0.0.1:8000/ws/test-floor";

  if (/\.vercel\.app$/i.test(window.location.hostname)) {
    return DEFAULT_PROD_WS;
  }

  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/test-floor`;
}

const STALE_MS = Number(process.env.NEXT_PUBLIC_STALE_TELEMETRY_MS ?? 45_000);

interface WsEnvelope {
  kind: string;
  event?: WaferTelemetryEvent | null;
  test_event?: import("@/types/events").TestEvent | null;
  status?: string;
  stream_meta?: { duplicate?: boolean; sequence_gap?: boolean };
}

/**
 * Authenticated WebSocket: die patches, live TestEvents, connection states.
 * Pause mode freezes UI updates without inventing replacement metrics.
 */
export function useWaferRealtime(waferId: string | null | undefined, enabled = true) {
  const queryClient = useQueryClient();
  const token = useAuthStore((s) => s.accessToken);
  const streamMode = useOpsStore((s) => s.streamMode);
  const reconnectNonce = useOpsStore((s) => s.reconnectNonce);
  const applyTelemetryEvent = useWaferStore((s) => s.applyTelemetryEvent);
  const applyYieldUpdate = useWaferStore((s) => s.applyYieldUpdate);
  const setLifecycle = useWaferStore((s) => s.setLifecycle);
  const setStatus = useConnectionStore((s) => s.setStatus);
  const setError = useConnectionStore((s) => s.setError);
  const touchMessage = useConnectionStore((s) => s.touchMessage);
  const forceOffline = useConnectionStore((s) => s.forceOffline);
  const upsertLive = useTestEventStore((s) => s.upsertLive);
  const retryRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seenIdsRef = useRef<Set<string>>(new Set());
  const lastSeqRef = useRef<number | null>(null);
  const hadLiveRef = useRef(false);
  const pausedRef = useRef(false);

  useEffect(() => {
    pausedRef.current = streamMode === "PAUSED";
  }, [streamMode]);

  useEffect(() => {
    if (!enabled || !token || streamMode === "PAUSED") {
      forceOffline();
      setLifecycle("offline");
      return;
    }

    let ws: WebSocket | null = null;
    let closed = false;
    let staleTimer: ReturnType<typeof setTimeout> | null = null;
    let pingTimer: ReturnType<typeof setInterval> | null = null;

    const armStaleWatch = () => {
      if (staleTimer) clearTimeout(staleTimer);
      staleTimer = setTimeout(() => {
        const st = useConnectionStore.getState().status;
        if (st === "LIVE" || st === "DEGRADED") {
          setStatus("STALE");
        }
      }, STALE_MS);
    };

    const syncBackend = () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      void queryClient.invalidateQueries({ queryKey: ["test-events"] });
      void queryClient.invalidateQueries({ queryKey: ["maintenance"] });
      void queryClient.invalidateQueries({ queryKey: ["test-limits"] });
      void queryClient.invalidateQueries({ queryKey: ["wafer"] });
      void queryClient.invalidateQueries({ queryKey: ["kpis"] });
    };

    const connect = () => {
      if (closed) return;
      setStatus("RECONNECTING");
      const base = resolveWsBase();
      if (!base.startsWith("ws://") && !base.startsWith("wss://")) {
        setError("Invalid WebSocket URL (expected ws/wss)");
        setStatus("OFFLINE");
        return;
      }
      const url = `${base}?token=${encodeURIComponent(token)}`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("LIVE");
        setError(null);
        if (waferId) setLifecycle("live");
        armStaleWatch();
        if (hadLiveRef.current) {
          syncBackend();
        }
        hadLiveRef.current = true;
        if (pingTimer) clearInterval(pingTimer);
        pingTimer = setInterval(() => {
          try {
            ws?.send(JSON.stringify({ kind: "ping" }));
          } catch {
            /* ignore */
          }
        }, 20_000);
      };

      ws.onmessage = (ev) => {
        if (pausedRef.current) return;
        try {
          const msg = JSON.parse(String(ev.data)) as WsEnvelope;
          if (msg.kind === "heartbeat" || msg.kind === "pong" || msg.kind === "projection_snapshot") {
            const st = useConnectionStore.getState().status;
            if (st === "RECONNECTING" || st === "OFFLINE") setStatus("LIVE");
            return;
          }
          if (msg.kind !== "telemetry_event" || !msg.event) return;

          const eventId = msg.event.event_id;
          if (eventId && seenIdsRef.current.has(eventId)) return;
          if (eventId) {
            seenIdsRef.current.add(eventId);
            if (seenIdsRef.current.size > 3000) {
              seenIdsRef.current = new Set([...seenIdsRef.current].slice(-1500));
            }
          }

          const seq = msg.event.sequence_number;
          if (typeof seq === "number") {
            lastSeqRef.current =
              lastSeqRef.current == null ? seq : Math.max(lastSeqRef.current, seq);
            touchMessage(seq);
          } else {
            touchMessage();
          }

          armStaleWatch();
          if (msg.stream_meta?.sequence_gap) {
            setStatus("DEGRADED");
          } else if (useConnectionStore.getState().status !== "LIVE") {
            setStatus("LIVE");
          }

          const floorEvent = testEventFromWsEnvelope({
            test_event: msg.test_event,
            event: msg.event as unknown as TelemetryEvent,
          });
          if (floorEvent) {
            upsertLive(floorEvent);
            void queryClient.invalidateQueries({ queryKey: ["test-events"] });
          }

          const event = msg.event;

          if (DIE_EVENT_TYPES.includes(event.event_type as (typeof DIE_EVENT_TYPES)[number])) {
            if (!waferId || !event.wafer_id || event.wafer_id === waferId) {
              applyTelemetryEvent(event);
            }
            return;
          }

          if (
            event.event_type === "yield_updated" ||
            event.event_type === "wafer_progress" ||
            event.event_type === "lot_completed"
          ) {
            if (!waferId || !event.wafer_id || event.wafer_id === waferId) {
              applyYieldUpdate(event);
            }
            void queryClient.invalidateQueries({ queryKey: ["wafer", waferId] });
          }

          if ((KPI_LIVE_EVENTS as readonly string[]).includes(event.event_type)) {
            if (useConnectionStore.getState().status === "LIVE") {
              void queryClient.invalidateQueries({ queryKey: ["kpis"] });
            }
            const updates = event.payload?.kpi_updates;
            if (updates && typeof updates === "object" && useConnectionStore.getState().status === "LIVE") {
              for (const id of Object.keys(updates as Record<string, unknown>)) {
                void queryClient.invalidateQueries({ queryKey: ["kpis", id] });
                void queryClient.invalidateQueries({ queryKey: ["kpis", id, "history"] });
              }
            }
          }

          if (event.event_type === "predictive_maintenance") {
            void queryClient.invalidateQueries({ queryKey: ["maintenance"] });
            if (event.tester_id) {
              void queryClient.invalidateQueries({ queryKey: ["maintenance", event.tester_id] });
            }
          }

          if (event.event_type === "dynamic_limit_updated") {
            void queryClient.invalidateQueries({ queryKey: ["test-limits"] });
          }

          void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
        } catch {
          /* ignore malformed frames */
        }
      };

      ws.onerror = () => {
        setError("WebSocket error");
      };

      ws.onclose = (ev) => {
        setLifecycle("offline");
        if (pingTimer) clearInterval(pingTimer);
        if (closed) {
          setStatus("OFFLINE");
          return;
        }
        if (ev.code === 4401 || ev.code === 4403) {
          // Do not clear the JWT here — WS auth can fail while REST session is still valid
          // (proxy blips, cold start). REST 401 handling is the source of truth for logout.
          setError(ev.code === 4401 ? "WebSocket unauthorized" : "WebSocket forbidden");
          setStatus("OFFLINE");
          return;
        }
        setStatus("RECONNECTING");
        const delay = Math.min(10_000, 1000 * 2 ** retryRef.current);
        retryRef.current += 1;
        timerRef.current = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      closed = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      if (staleTimer) clearTimeout(staleTimer);
      if (pingTimer) clearInterval(pingTimer);
      ws?.close();
    };
  }, [
    enabled,
    token,
    waferId,
    streamMode,
    reconnectNonce,
    queryClient,
    applyTelemetryEvent,
    applyYieldUpdate,
    setLifecycle,
    setStatus,
    setError,
    touchMessage,
    forceOffline,
    upsertLive,
  ]);
}
