import { createContext } from "react";
import type { AppContextValue } from "./contextTypes";

export const AppContext = createContext<AppContextValue | null>(null);
