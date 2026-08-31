import { useEffect } from "react";
import { ApiError } from "@/api/client";
import { getLotDieParameters, getLotDies, getLots } from "@/api/endpoints";
import { useAppContext } from "@/state/useAppContext";

export function useSelectorData() {
  const { state, dispatch } = useAppContext();

  useEffect(() => {
    let cancelled = false;
    async function loadLots() {
      dispatch({ type: "SET_SELECTORS_LOADING", payload: { lots: true } });
      try {
        const res = await getLots();
        if (!cancelled) {
          dispatch({ type: "SET_AVAILABLE_LOTS", payload: res.lots });
        }
      } catch (err) {
        if (!cancelled) {
          if (err instanceof ApiError) {
            dispatch({
              type: "SET_ERROR",
              payload: { message: err.message, code: err.code },
            });
          } else {
            dispatch({
              type: "SET_ERROR",
              payload: { message: "Unable to load lots from service." },
            });
          }
        }
      } finally {
        if (!cancelled) {
          dispatch({ type: "SET_SELECTORS_LOADING", payload: { lots: false } });
        }
      }
    }
    void loadLots();
    return () => {
      cancelled = true;
    };
  }, [dispatch]);

  useEffect(() => {
    let cancelled = false;
    if (!state.selectedLot) {
      return;
    }
    async function loadDies() {
      dispatch({ type: "SET_SELECTORS_LOADING", payload: { dies: true } });
      try {
        const res = await getLotDies(state.selectedLot);
        if (!cancelled) {
          dispatch({ type: "SET_AVAILABLE_DIES", payload: res.dies });
        }
      } catch (err) {
        if (!cancelled) {
          if (err instanceof ApiError) {
            dispatch({
              type: "SET_ERROR",
              payload: { message: err.message, code: err.code },
            });
          } else {
            dispatch({
              type: "SET_ERROR",
              payload: { message: "Unable to load dies for selected lot." },
            });
          }
        }
      } finally {
        if (!cancelled) {
          dispatch({ type: "SET_SELECTORS_LOADING", payload: { dies: false } });
        }
      }
    }
    void loadDies();
    return () => {
      cancelled = true;
    };
  }, [dispatch, state.selectedLot]);

  useEffect(() => {
    let cancelled = false;
    if (!state.selectedLot || !state.selectedDie) {
      return;
    }
    async function loadParameters() {
      dispatch({ type: "SET_SELECTORS_LOADING", payload: { parameters: true } });
      try {
        const res = await getLotDieParameters(state.selectedLot, state.selectedDie);
        if (!cancelled) {
          dispatch({ type: "SET_AVAILABLE_PARAMETERS", payload: res.parameters });
        }
      } catch (err) {
        if (!cancelled) {
          if (err instanceof ApiError) {
            dispatch({
              type: "SET_ERROR",
              payload: { message: err.message, code: err.code },
            });
          } else {
            dispatch({
              type: "SET_ERROR",
              payload: { message: "Unable to load parameters for selected die." },
            });
          }
        }
      } finally {
        if (!cancelled) {
          dispatch({ type: "SET_SELECTORS_LOADING", payload: { parameters: false } });
        }
      }
    }
    void loadParameters();
    return () => {
      cancelled = true;
    };
  }, [dispatch, state.selectedDie, state.selectedLot]);
}
