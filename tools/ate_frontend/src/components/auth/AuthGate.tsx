"use client";

import { useEffect, useState } from "react";
import { VerilumenBrand } from "@/components/branding/VerilumenBrand";
import { fetchMe, login } from "@/services/api";
import { useAuthStore } from "@/stores/authStore";
import type { AppRole } from "@/stores/authStore";

function isUnauthorizedError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /unauthorized|401|invalid or expired token|not authenticated/i.test(msg);
}

/**
 * Ensures a JWT session before rendering the dashboard.
 * Shows the login form when there is no stored session (no silent auto-login).
 */
export function AuthGate({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((s) => s.accessToken);
  const setSession = useAuthStore((s) => s.setSession);
  const clearSession = useAuthStore((s) => s.clearSession);
  const [username, setUsername] = useState(
    process.env.NEXT_PUBLIC_DEMO_USER ?? "viewer",
  );
  const [password, setPassword] = useState(
    process.env.NEXT_PUBLIC_DEMO_PASSWORD ?? "viewer123",
  );
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [restoring, setRestoring] = useState(false);

  useEffect(() => {
    const unsub = useAuthStore.persist.onFinishHydration(() => setHydrated(true));
    setHydrated(useAuthStore.persist.hasHydrated());
    return unsub;
  }, []);

  // Sign out clears the token — return to the login form.
  useEffect(() => {
    if (!token) {
      setReady(false);
      setRestoring(false);
    }
  }, [token]);

  useEffect(() => {
    if (!hydrated) return;
    let cancelled = false;
    (async () => {
      const current = useAuthStore.getState().accessToken;
      if (!current) {
        setReady(false);
        setRestoring(false);
        return;
      }

      setRestoring(true);
      try {
        const me = await fetchMe();
        if (cancelled) return;
        setSession({
          accessToken: current,
          username: me.username,
          role: me.role as AppRole,
          permissions: me.permissions,
        });
        setReady(true);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        // Only wipe the stored JWT when the server rejects it.
        // Network / cold-start failures must not force re-login after refresh.
        if (isUnauthorizedError(err)) {
          clearSession();
          setReady(false);
          setError("Session expired. Please sign in again.");
        } else {
          setReady(true);
          setError(null);
        }
      } finally {
        if (!cancelled) setRestoring(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated]);

  if (!hydrated || restoring) {
    return (
      <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
        <div className="vl-enter mb-6">
          <VerilumenBrand size="auth" />
        </div>
        <p className="text-[12px] text-[var(--muted)]">
          {restoring ? "Restoring session…" : "Loading…"}
        </p>
      </div>
    );
  }

  if (ready && token) return <>{children}</>;

  // After Sign out, reset ready so the next successful login can enter the dashboard.
  // (token is already null from clearSession)

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
      <div className="vl-enter mb-6">
        <VerilumenBrand size="auth" />
      </div>
      <p className="vl-enter vl-enter-delay-1 mb-5 text-[12px] leading-relaxed text-[var(--muted)]">
        Sign in with an authorized engineering role. Unauthorized API / WebSocket
        access is rejected.
      </p>
      <form
        className="vl-surface vl-enter vl-enter-delay-2 flex flex-col gap-3.5 p-5"
        onSubmit={(e) => {
          e.preventDefault();
          void (async () => {
            setBusy(true);
            setError(null);
            try {
              const res = await login(username, password);
              setSession({
                accessToken: res.access_token,
                username: res.username,
                role: res.role as AppRole,
              });
              try {
                const me = await fetchMe();
                setSession({
                  accessToken: res.access_token,
                  username: me.username,
                  role: me.role as AppRole,
                  permissions: me.permissions,
                });
              } catch (meErr) {
                // Token was issued; only fail the login UI if /auth/me rejects it.
                if (isUnauthorizedError(meErr)) {
                  clearSession();
                  throw meErr;
                }
              }
              setReady(true);
            } catch (err) {
              setError(err instanceof Error ? err.message : "Login failed");
              setReady(false);
            } finally {
              setBusy(false);
            }
          })();
        }}
      >
        <label className="vl-label">
          Username
          <input
            className="vl-field mt-1.5 w-full px-3 py-2 text-[12px]"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
          />
        </label>
        <label className="vl-label">
          Password
          <input
            type="password"
            className="vl-field mt-1.5 w-full px-3 py-2 text-[12px]"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
          />
        </label>
        {error ? <div className="text-[11px] text-[var(--red)]">{error}</div> : null}
        <button
          type="submit"
          disabled={busy}
          className="mt-1 rounded-[6px] border border-[rgba(107,193,242,0.55)] bg-[linear-gradient(180deg,rgba(107,193,242,0.12),transparent)] px-3 py-2 text-[12px] font-semibold text-[var(--cyan)] transition-colors hover:border-[var(--cyan)] disabled:opacity-40"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <div className="text-[10px] text-[var(--muted-2)]">
          Roles: viewer / test_eng / process_eng / ai_eng / maint_eng / admin
        </div>
      </form>
    </div>
  );
}
