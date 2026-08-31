import type { Dispatch } from "react";
import type { AppAction, AppState } from "./types";

export interface AppContextValue {
  state: AppState;
  dispatch: Dispatch<AppAction>;
}
