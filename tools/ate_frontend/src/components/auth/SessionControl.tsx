"use client";

import { LogOut } from "lucide-react";
import { useAuthStore } from "@/stores/authStore";

/**
 * Shows current user + Sign out so engineers can switch roles (e.g. viewer → test_eng).
 */
export function SessionControl() {
  const username = useAuthStore((s) => s.username);
  const role = useAuthStore((s) => s.role);
  const clearSession = useAuthStore((s) => s.clearSession);

  return (
    <div className="flex items-center gap-2 border-l border-[var(--line)] pl-3">
      <div className="text-right leading-tight">
        <div className="font-mono text-[11px] font-semibold text-[var(--text)]">
          {username ?? "—"}
        </div>
        <div className="text-[9px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          {role?.replaceAll("_", " ") ?? "signed in"}
        </div>
      </div>
      <button
        type="button"
        title="Sign out"
        onClick={() => clearSession()}
        className="inline-flex h-9 items-center gap-1.5 rounded-[6px] border border-[var(--line-bright)] bg-[linear-gradient(180deg,rgba(255,255,255,0.04),transparent),var(--panel)] px-2.5 text-[11px] font-semibold text-[var(--muted)] transition-colors hover:border-[rgba(107,193,242,0.55)] hover:text-[var(--cyan)]"
        aria-label="Sign out"
      >
        <LogOut className="h-3.5 w-3.5" />
        Sign out
      </button>
    </div>
  );
}
