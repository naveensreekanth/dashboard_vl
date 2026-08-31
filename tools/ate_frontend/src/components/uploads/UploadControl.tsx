"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Button } from "@/components/ui/button";
import { uploadFloorFile, uploadShmooFile } from "@/services/api";
import { useAuthStore } from "@/stores/authStore";
import { useKpiStore } from "@/stores/kpiStore";
import { useShmooStore } from "@/stores/shmooStore";

type UploadKind = "auto" | "wafer_image" | "stil" | "stdf" | "log" | "shmoo";

function formatUploadError(err: unknown): string {
  if (!(err instanceof Error)) return "Upload failed";
  const msg = err.message;
  try {
    const jsonMatch = msg.match(/\{.*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]) as { detail?: unknown };
      if (typeof parsed.detail === "string") return parsed.detail;
      if (Array.isArray(parsed.detail)) {
        return parsed.detail
          .map((d) =>
            typeof d === "object" && d && "msg" in d
              ? String((d as { msg: string }).msg)
              : String(d),
          )
          .join("; ");
      }
    }
  } catch {
    /* keep raw */
  }
  return msg;
}

function scrollToSection(id: string) {
  const run = () => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  // Wait for React to paint the results panel, then scroll twice for reliability.
  requestAnimationFrame(() => {
    run();
    window.setTimeout(run, 120);
  });
}

/**
 * Header upload control — portals popover so floor controls cannot cover it.
 * Keeps the panel open after success so the uploaded filename stays visible.
 */
export function UploadControl() {
  const inputRef = useRef<HTMLInputElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const queryClient = useQueryClient();
  const role = useAuthStore((s) => s.role);
  const canUpload = useAuthStore((s) => s.hasPermission("write:telemetry"));
  const setShmooSession = useShmooStore((s) => s.setSession);
  const selectKpi = useKpiStore((s) => s.selectKpi);

  const [open, setOpen] = useState(false);
  const [kind, setKind] = useState<UploadKind>("auto");
  const [busy, setBusy] = useState(false);
  const [pickedName, setPickedName] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [menuPos, setMenuPos] = useState<{ top: number; right: number }>({
    top: 0,
    right: 16,
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open || !triggerRef.current) return;
    const update = () => {
      const rect = triggerRef.current!.getBoundingClientRect();
      setMenuPos({
        top: rect.bottom + 8,
        right: Math.max(12, window.innerWidth - rect.right),
      });
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open]);

  const onPick = () => {
    if (!canUpload) {
      setError(
        `Your role (${role ?? "unknown"}) cannot upload. Sign in as test_eng / admin / process_eng.`,
      );
      return;
    }
    inputRef.current?.click();
  };

  const onFile = async (file: File | null) => {
    if (!file) return;
    if (!canUpload) {
      setError(
        `Your role (${role ?? "unknown"}) cannot upload. Sign in as test_eng / admin / process_eng.`,
      );
      return;
    }
    setBusy(true);
    setError(null);
    setMessage(null);
    setPickedName(file.name);
    try {
      const isShmoo =
        kind === "shmoo" ||
        (kind === "auto" && /\.(csv|xlsx|xls)$/i.test(file.name));

      if (isShmoo) {
        const res = await uploadShmooFile(file);
        setShmooSession({
          sessionId: res.session_id,
          filename: res.filename,
          meta: res.meta,
          results: res.results,
          plotUrl: res.plot_url,
          plotUrls: {
            character: res.plot_urls?.character ?? res.plot_url,
            yield: res.plot_urls?.yield ?? null,
            debug: res.plot_urls?.debug ?? null,
          },
        });
        setMessage(
          `Uploaded ${res.filename ?? file.name} · CV ${(res.results.cv_accuracy * 100).toFixed(1)}% · open SHMOO KPI`,
        );
        void queryClient.invalidateQueries({ queryKey: ["kpis"] });
        void queryClient.invalidateQueries({ queryKey: ["shmoo", "latest"] });
        // Keep popover open so the filename/success stay visible.
        scrollToSection("optimization-parameters");
        selectKpi("m_bist_shmoo");
      } else {
        const res = await uploadFloorFile(file, kind);
        const detail =
          res.kind === "wafer_image"
            ? `Uploaded ${file.name} · wafer image · ${res.dies ?? 0} dies`
            : `Uploaded ${file.name} · ${String(res.kind).toUpperCase()} · ${res.events_accepted ?? 0} events`;
        setMessage(detail);
        void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
        void queryClient.invalidateQueries({ queryKey: ["wafer"] });
        void queryClient.invalidateQueries({ queryKey: ["test-events"] });
        void queryClient.invalidateQueries({ queryKey: ["kpis"] });
        scrollToSection("live-wafer-map");
      }
    } catch (err) {
      setError(formatUploadError(err));
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const accept =
    kind === "wafer_image"
      ? ".png,.jpg,.jpeg,.bmp,.webp"
      : kind === "stil"
        ? ".stil,.stilt,.txt"
        : kind === "stdf"
          ? ".stdf,.std,.atr,.txt"
          : kind === "log"
            ? ".log,.txt,.csv"
            : kind === "shmoo"
              ? ".csv,.xlsx,.xls"
              : ".png,.jpg,.jpeg,.bmp,.webp,.stil,.stdf,.std,.log,.txt,.csv,.xlsx,.xls";

  const panel = open ? (
    <div
      data-upload-popover="1"
      className="fixed z-[200] w-[340px] rounded-[8px] border border-[var(--line-bright)] p-4 shadow-[0_16px_48px_rgba(0,0,0,0.65)]"
      style={{
        top: menuPos.top,
        right: menuPos.right,
        background: "var(--panel-elevated)",
      }}
    >
      <div className="vl-label mb-2">Upload artifacts</div>
      <p className="mb-2 text-[11px] leading-relaxed text-[var(--muted)]">
        Floor: wafer image, STDF/STIL, or test log. Optimization: Shmoo CSV/XLSX (VDD × Frequency).
      </p>
      <p className="mb-3 rounded border border-[var(--line)] bg-[var(--panel)] px-2.5 py-2 text-[10px] leading-relaxed text-[var(--muted-2)]">
        <span className="text-[var(--cyan)]">Auto tip:</span> CSV/XLSX opens SHMOO ML KPI;
        images/logs go to wafer map
      </p>

      {!canUpload ? (
        <p className="mb-3 rounded border border-[var(--amber)]/40 bg-[var(--amber-dim)] px-2.5 py-2 text-[11px] leading-relaxed text-[var(--amber)]">
          Signed in as <span className="font-mono">{role ?? "—"}</span> — cannot upload. Use{" "}
          <span className="font-mono">test_eng</span> / <span className="font-mono">test123</span> or{" "}
          <span className="font-mono">admin</span> / <span className="font-mono">admin123</span>.
        </p>
      ) : (
        <p className="mb-3 text-[10px] text-[var(--muted-2)]">
          Role <span className="font-mono text-[var(--cyan)]">{role}</span> · upload enabled
        </p>
      )}

      <label className="mb-2 flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
        File type
        <select
          value={kind}
          disabled={!canUpload || busy}
          onChange={(e) => setKind(e.target.value as UploadKind)}
          className="vl-field px-2.5 py-1.5 text-[11.5px] normal-case tracking-normal disabled:opacity-50"
        >
          <option value="auto">Auto-detect</option>
          <option value="wafer_image">Wafer image</option>
          <option value="stil">STIL</option>
          <option value="stdf">STDF</option>
          <option value="log">Test log</option>
          <option value="shmoo">Shmoo dataset</option>
        </select>
      </label>

      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={accept}
        disabled={!canUpload || busy}
        onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
      />

      {pickedName ? (
        <div className="mb-2 rounded border border-[var(--line)] bg-[var(--panel)] px-2.5 py-2 font-mono text-[11px] text-[var(--text)]">
          File: <span className="text-[var(--cyan)]">{pickedName}</span>
        </div>
      ) : null}

      <div className="mt-2 flex gap-2">
        <Button type="button" disabled={busy} onClick={onPick}>
          {busy ? "Uploading…" : canUpload ? "Choose file" : "Need engineer login"}
        </Button>
        <Button type="button" variant="ghost" onClick={() => setOpen(false)}>
          Close
        </Button>
      </div>

      {message ? (
        <div className="mt-3 space-y-2">
          <div className="rounded border border-[var(--green)]/40 bg-[var(--green-dim)] px-2.5 py-2 text-[11px] leading-relaxed text-[var(--green)]">
            {message}
          </div>
          <Button
            type="button"
            onClick={() => {
              setOpen(false);
              const isShmooResult =
                message.toLowerCase().includes("shmoo") ||
                message.toLowerCase().includes("cv ") ||
                message.toLowerCase().includes("m-bist");
              if (isShmooResult) {
                scrollToSection("optimization-parameters");
                selectKpi("m_bist_shmoo");
              } else {
                scrollToSection("live-wafer-map");
              }
            }}
          >
            View results
          </Button>
        </div>
      ) : null}
      {error ? <div className="mt-2 text-[11px] text-[var(--red)]">{error}</div> : null}
    </div>
  ) : null;

  return (
    <div className="relative z-[60]">
      <button
        ref={triggerRef}
        type="button"
        title={
          canUpload
            ? "Upload wafer / STDF·STIL / log / Shmoo"
            : "Upload (requires engineer role)"
        }
        onClick={() => setOpen((v) => !v)}
        className="inline-flex h-9 w-9 items-center justify-center rounded-[6px] border border-[var(--line-bright)] bg-[linear-gradient(180deg,rgba(255,255,255,0.04),transparent),var(--panel)] text-[var(--cyan)] transition-colors hover:border-[rgba(107,193,242,0.55)]"
        aria-label="Upload files"
        aria-expanded={open}
      >
        <Upload className="h-4 w-4" />
      </button>

      {mounted && panel ? createPortal(panel, document.body) : null}
    </div>
  );
}
