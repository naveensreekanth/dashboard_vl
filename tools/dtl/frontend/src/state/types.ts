import type { LotRecommendationResult } from "@/api/types";

export interface AppState {
  selectedLot: string;
  selectedDie: string;
  selectedParameter: string;
  /** Empty string = legacy (production_month omitted). */
  selectedProductionMonth: string;
  availableLots: string[];
  availableDies: string[];
  availableParameters: string[];
  selectorsLoading: {
    lots: boolean;
    dies: boolean;
    parameters: boolean;
  };
  recommendation: LotRecommendationResult | null;
  loading: boolean;
  error: string | null;
  errorCode: string | null;
  serviceReady: boolean | null;
  serviceReason: string | null;
}

export type AppAction =
  | { type: "SET_LOT"; payload: string }
  | { type: "SET_DIE"; payload: string }
  | { type: "SET_PARAMETER"; payload: string }
  | { type: "SET_PRODUCTION_MONTH"; payload: string }
  | { type: "SET_AVAILABLE_LOTS"; payload: string[] }
  | { type: "SET_AVAILABLE_DIES"; payload: string[] }
  | { type: "SET_AVAILABLE_PARAMETERS"; payload: string[] }
  | {
      type: "SET_SELECTORS_LOADING";
      payload: Partial<{ lots: boolean; dies: boolean; parameters: boolean }>;
    }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_RECOMMENDATION"; payload: LotRecommendationResult }
  | { type: "SET_ERROR"; payload: { message: string; code?: string | null } }
  | { type: "CLEAR_ERROR" }
  | { type: "SET_SERVICE_STATUS"; payload: { ready: boolean; reason?: string | null } };

export const initialState: AppState = {
  selectedLot: "",
  selectedDie: "",
  selectedParameter: "",
  selectedProductionMonth: "",
  availableLots: [],
  availableDies: [],
  availableParameters: [],
  selectorsLoading: {
    lots: false,
    dies: false,
    parameters: false,
  },
  recommendation: null,
  loading: false,
  error: null,
  errorCode: null,
  serviceReady: null,
  serviceReason: null,
};

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_LOT":
      return {
        ...state,
        selectedLot: action.payload,
        selectedDie: "",
        selectedParameter: "",
        availableDies: [],
        availableParameters: [],
        recommendation: null,
      };
    case "SET_DIE":
      return {
        ...state,
        selectedDie: action.payload,
        selectedParameter: "",
        availableParameters: [],
        recommendation: null,
      };
    case "SET_PARAMETER":
      return { ...state, selectedParameter: action.payload, recommendation: null };
    case "SET_PRODUCTION_MONTH":
      return { ...state, selectedProductionMonth: action.payload, recommendation: null };
    case "SET_AVAILABLE_LOTS":
      return { ...state, availableLots: action.payload };
    case "SET_AVAILABLE_DIES":
      return { ...state, availableDies: action.payload };
    case "SET_AVAILABLE_PARAMETERS":
      return {
        ...state,
        availableParameters: action.payload,
        selectedParameter:
          action.payload.includes(state.selectedParameter) && state.selectedParameter
            ? state.selectedParameter
            : (action.payload[0] ?? ""),
      };
    case "SET_SELECTORS_LOADING":
      return {
        ...state,
        selectorsLoading: {
          ...state.selectorsLoading,
          ...action.payload,
        },
      };
    case "SET_LOADING":
      return { ...state, loading: action.payload, error: null, errorCode: null };
    case "SET_RECOMMENDATION":
      return {
        ...state,
        recommendation: action.payload,
        loading: false,
        error: null,
        errorCode: null,
      };
    case "SET_ERROR":
      return {
        ...state,
        loading: false,
        error: action.payload.message,
        errorCode: action.payload.code ?? null,
      };
    case "CLEAR_ERROR":
      return { ...state, error: null, errorCode: null };
    case "SET_SERVICE_STATUS":
      return {
        ...state,
        serviceReady: action.payload.ready,
        serviceReason: action.payload.reason ?? null,
      };
    default:
      return state;
  }
}
