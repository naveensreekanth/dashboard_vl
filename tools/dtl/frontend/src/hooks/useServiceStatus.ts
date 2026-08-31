import { useEffect } from "react";
import { getReady } from "@/api/endpoints";
import { useAppContext } from "@/state/useAppContext";

export function useServiceStatus(pollMs = 10000) {
  const { dispatch } = useAppContext();

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const ready = await getReady();
        if (!cancelled) {
          dispatch({
            type: "SET_SERVICE_STATUS",
            payload: {
              ready: ready.status === "ready",
              reason: ready.reason ?? null,
            },
          });
        }
      } catch {
        if (!cancelled) {
          dispatch({
            type: "SET_SERVICE_STATUS",
            payload: { ready: false, reason: "SERVICE_UNAVAILABLE" },
          });
        }
      }
    }

    void check();
    const id = window.setInterval(() => void check(), pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [dispatch, pollMs]);
}
