import { useCallback, useEffect, useState } from "react";
import { ApiError } from "@/api/client";
import {
  getDieConditions,
  getDieDistribution,
  getDieMeasurements,
} from "@/api/endpoints";
import type {
  DieConditionsResponse,
  DieDistributionResponse,
  DieMeasurementResponse,
} from "@/api/types";
import { useAppContext } from "@/state/useAppContext";

export interface MeasurementPanelState {
  measurement: DieMeasurementResponse | null;
  distribution: DieDistributionResponse | null;
  conditions: DieConditionsResponse | null;
  loading: {
    measurement: boolean;
    distribution: boolean;
    conditions: boolean;
  };
  error: {
    measurement: string | null;
    distribution: string | null;
    conditions: string | null;
  };
  selectionKey: string | null;
  retry: () => void;
}

const emptyErrors = {
  measurement: null as string | null,
  distribution: null as string | null,
  conditions: null as string | null,
};

const idleLoading = {
  measurement: false,
  distribution: false,
  conditions: false,
};

function userFacingError(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    return err.message;
  }
  return fallback;
}

function isAbortError(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === "AbortError") ||
    (err instanceof Error && err.name === "AbortError")
  );
}

export function useMeasurementData(): MeasurementPanelState {
  const { state } = useAppContext();
  const { selectedLot, selectedDie, selectedParameter } = state;

  const [measurement, setMeasurement] = useState<DieMeasurementResponse | null>(null);
  const [distribution, setDistribution] = useState<DieDistributionResponse | null>(null);
  const [conditions, setConditions] = useState<DieConditionsResponse | null>(null);
  const [loading, setLoading] = useState(idleLoading);
  const [error, setError] = useState(emptyErrors);
  const [selectionKey, setSelectionKey] = useState<string | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);

  const retry = useCallback(() => {
    setRetryNonce((n) => n + 1);
  }, []);

  useEffect(() => {
    if (!selectedLot || !selectedDie || !selectedParameter) {
      setMeasurement(null);
      setDistribution(null);
      setConditions(null);
      setLoading(idleLoading);
      setError(emptyErrors);
      setSelectionKey(null);
      return;
    }

    const key = `${selectedLot}|${selectedDie}|${selectedParameter}`;
    const controller = new AbortController();
    let cancelled = false;

    setSelectionKey(key);
    setMeasurement(null);
    setDistribution(null);
    setConditions(null);
    setError(emptyErrors);
    setLoading({
      measurement: true,
      distribution: true,
      conditions: true,
    });

    const params = { lot_id: selectedLot, parameter: selectedParameter };

    void (async () => {
      try {
        const data = await getDieMeasurements(selectedDie, params, controller.signal);
        if (cancelled) return;
        setMeasurement(data);
      } catch (err) {
        if (cancelled || isAbortError(err)) return;
        setMeasurement(null);
        setError((prev) => ({
          ...prev,
          measurement: userFacingError(err, "Unable to load measurement data."),
        }));
      } finally {
        if (!cancelled) {
          setLoading((prev) => ({ ...prev, measurement: false }));
        }
      }
    })();

    void (async () => {
      try {
        const data = await getDieDistribution(
          selectedDie,
          { ...params, scope: "die" },
          controller.signal,
        );
        if (cancelled) return;
        setDistribution(data);
      } catch (err) {
        if (cancelled || isAbortError(err)) return;
        setDistribution(null);
        setError((prev) => ({
          ...prev,
          distribution: userFacingError(err, "Unable to load distribution data."),
        }));
      } finally {
        if (!cancelled) {
          setLoading((prev) => ({ ...prev, distribution: false }));
        }
      }
    })();

    void (async () => {
      try {
        const data = await getDieConditions(selectedDie, params, controller.signal);
        if (cancelled) return;
        setConditions(data);
      } catch (err) {
        if (cancelled || isAbortError(err)) return;
        setConditions(null);
        setError((prev) => ({
          ...prev,
          conditions: userFacingError(err, "Unable to load condition data."),
        }));
      } finally {
        if (!cancelled) {
          setLoading((prev) => ({ ...prev, conditions: false }));
        }
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedDie, selectedLot, selectedParameter, retryNonce]);

  return {
    measurement,
    distribution,
    conditions,
    loading,
    error,
    selectionKey,
    retry,
  };
}
