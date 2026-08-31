import { useContext } from "react";
import { AppContext } from "./context";
import type { AppContextValue } from "./contextTypes";

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return ctx;
}
