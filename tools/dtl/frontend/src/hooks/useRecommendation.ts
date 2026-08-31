import { useCallback } from "react";
import { ApiError } from "@/api/client";
import { postRecommendation } from "@/api/endpoints";
import { useAppContext } from "@/state/useAppContext";

export function useRecommendation() {
  const { state, dispatch } = useAppContext();

  const fetchRecommendation = useCallback(async () => {
    if (!state.selectedLot || !state.selectedDie || !state.selectedParameter) {
      dispatch({
        type: "SET_ERROR",
        payload: {
          message: "Lot, Die, and Parameter selection are required.",
          code: "VALIDATION_ERROR",
        },
      });
      return;
    }

    dispatch({ type: "SET_LOADING", payload: true });
    try {
      const result = await postRecommendation({
        lot_id: state.selectedLot,
        die_id: state.selectedDie,
        parameters: [state.selectedParameter],
        ...(state.selectedProductionMonth
          ? { production_month: state.selectedProductionMonth }
          : {}),
      });
      dispatch({ type: "SET_RECOMMENDATION", payload: result });
    } catch (err) {
      if (err instanceof ApiError) {
        dispatch({
          type: "SET_ERROR",
          payload: { message: err.message, code: err.code },
        });
      } else if (err instanceof Error) {
        dispatch({ type: "SET_ERROR", payload: { message: err.message } });
      } else {
        dispatch({
          type: "SET_ERROR",
          payload: { message: "Recommendation service could not process the request." },
        });
      }
    }
  }, [
    dispatch,
    state.selectedDie,
    state.selectedLot,
    state.selectedParameter,
    state.selectedProductionMonth,
  ]);

  return { fetchRecommendation };
}
