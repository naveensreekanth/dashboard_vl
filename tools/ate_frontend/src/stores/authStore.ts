import { create } from "zustand";
import { persist } from "zustand/middleware";

export type AppRole =
  | "VIEWER"
  | "TEST_ENGINEER"
  | "PROCESS_ENGINEER"
  | "AI_ENGINEER"
  | "MAINTENANCE_ENGINEER"
  | "ADMIN";

/** Mirrors backend `ROLE_PERMISSIONS` so UI gates work even if /auth/me permissions are empty. */
const ROLE_PERMISSIONS: Record<AppRole, readonly string[]> = {
  VIEWER: [
    "read:dashboard",
    "read:wafer",
    "read:events",
    "read:kpis",
    "read:maintenance",
    "read:limits",
    "read:aggregations",
    "stream:ws",
  ],
  TEST_ENGINEER: [
    "read:dashboard",
    "read:wafer",
    "read:events",
    "read:kpis",
    "read:maintenance",
    "read:limits",
    "read:aggregations",
    "stream:ws",
    "write:events:ack",
    "write:telemetry",
    "read:telemetry",
  ],
  PROCESS_ENGINEER: [
    "read:dashboard",
    "read:wafer",
    "read:events",
    "read:kpis",
    "read:maintenance",
    "read:limits",
    "read:aggregations",
    "stream:ws",
    "write:events:ack",
    "write:telemetry",
    "read:telemetry",
    "write:limits:recommend",
    "write:limits:approve",
    "write:limits:reject",
    "write:limits:rollback",
  ],
  AI_ENGINEER: [
    "read:dashboard",
    "read:wafer",
    "read:events",
    "read:kpis",
    "read:maintenance",
    "read:limits",
    "read:aggregations",
    "stream:ws",
    "write:maintenance:predict",
    "write:kpis",
    "read:telemetry",
    "write:limits:recommend",
  ],
  MAINTENANCE_ENGINEER: [
    "read:dashboard",
    "read:wafer",
    "read:events",
    "read:kpis",
    "read:maintenance",
    "read:limits",
    "read:aggregations",
    "stream:ws",
    "write:maintenance:predict",
    "write:events:ack",
  ],
  ADMIN: [], // ADMIN is wildcard in hasPermission
};

export function permissionsForRole(role: AppRole | null | undefined): string[] {
  if (!role) return [];
  if (role === "ADMIN") return ["*"];
  return [...(ROLE_PERMISSIONS[role] ?? [])];
}

interface AuthState {
  accessToken: string | null;
  username: string | null;
  role: AppRole | null;
  permissions: string[];
  setSession: (s: {
    accessToken: string;
    username: string;
    role: AppRole;
    permissions?: string[];
  }) => void;
  clearSession: () => void;
  hasPermission: (p: string) => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      accessToken: null,
      username: null,
      role: null,
      permissions: [],
      setSession: ({ accessToken, username, role, permissions }) =>
        set({
          accessToken,
          username,
          role,
          // Never leave an engineer with an empty permission list (breaks UploadControl).
          permissions:
            permissions && permissions.length > 0
              ? permissions
              : permissionsForRole(role),
        }),
      clearSession: () =>
        set({ accessToken: null, username: null, role: null, permissions: [] }),
      hasPermission: (p) => {
        const role = get().role;
        if (role === "ADMIN") return true;
        const listed = get().permissions;
        if (listed.includes(p) || listed.includes("*")) return true;
        // Fallback to role map when persisted session predates permissions hydration.
        return permissionsForRole(role).includes(p);
      },
    }),
    { name: "wyi-auth" },
  ),
);
