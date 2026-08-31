import { useRef, useState } from "react";
import { ApiError } from "@/api/client";
import {
  postAnalysisUpload,
  type AnalysisUploadResult,
} from "@/api/endpoints";

type MonthSlot = "january" | "february" | "march";

const SLOTS: { key: MonthSlot; label: string; month: string }[] = [
  { key: "january", label: "January 2026", month: "2026-01" },
  { key: "february", label: "February 2026", month: "2026-02" },
  { key: "march", label: "March 2026", month: "2026-03" },
];

export interface UploadAnalysisPanelProps {
  onSessionReady: (result: AnalysisUploadResult) => void;
  analyzing?: boolean;
  jobStage?: string | null;
  progressPct?: number | null;
}

export function UploadAnalysisPanel({
  onSessionReady,
  analyzing = false,
  jobStage,
  progressPct,
}: UploadAnalysisPanelProps) {
  const januaryRef = useRef<HTMLInputElement>(null);
  const februaryRef = useRef<HTMLInputElement>(null);
  const marchRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<Partial<Record<MonthSlot, File>>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const allReady = Boolean(files.january && files.february && files.march);
  const busy = loading || analyzing;

  const onFileChange = (slot: MonthSlot, e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? undefined;
    setFiles((prev) => ({ ...prev, [slot]: f }));
    setError(null);
  };

  const onAnalyze = async () => {
    if (!files.january || !files.february || !files.march) {
      setError("Upload January, February, and March files before analyzing.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await postAnalysisUpload({
        january: files.january,
        february: files.february,
        march: files.march,
      });
      onSessionReady(result);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Upload analysis failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <section
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-5 shadow-sm transition-colors"
      data-testid="upload-analysis-panel"
      aria-label="Upload DTL test data for three-month analysis"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-3">
        <div>
          <h2 className="text-sm font-semibold tracking-wide text-[var(--text-primary)] uppercase">
            Upload DTL Test Data
          </h2>
          <p className="mt-0.5 text-xs text-[var(--text-muted)]">
            Upload test data for January, February, and March slots. All three slots are required.
          </p>
        </div>
        <span className="rounded border border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)] px-2 py-0.5 text-[10px] font-mono text-[var(--text-muted)] uppercase">
          Input Stage
        </span>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {SLOTS.map(({ key, label }) => {
          const inputRef =
            key === "january" ? januaryRef : key === "february" ? februaryRef : marchRef;
          const hasFile = Boolean(files[key]);
          return (
            <div
              key={key}
              className={`rounded-md border p-3 transition-colors ${
                hasFile
                  ? "border-emerald-500/40 bg-emerald-500/5 dark:bg-emerald-950/20"
                  : "border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)]"
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".csv,.zip"
                className="hidden"
                data-testid={`upload-${key}-input`}
                onChange={(e) => onFileChange(key, e)}
              />
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--text-primary)]">{label}</span>
                {hasFile ? (
                  <span className="text-[10px] text-emerald-500 font-mono font-medium">✓ Ready</span>
                ) : (
                  <span className="text-[10px] text-[var(--text-muted)] font-mono">Required</span>
                )}
              </div>

              <div className="mt-3 flex items-center gap-2">
                <button
                  type="button"
                  className="rounded border border-[var(--border-muted)] bg-[var(--bg-panel)] px-2.5 py-1.5 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors disabled:opacity-50"
                  data-testid={`upload-${key}-choose`}
                  onClick={() => inputRef.current?.click()}
                  disabled={busy}
                >
                  Choose File
                </button>
                <span
                  className="truncate text-[11px] font-mono text-[var(--text-muted)] flex-1"
                  data-testid={`upload-${key}-filename`}
                  title={files[key]?.name ?? "No file chosen"}
                >
                  {files[key]?.name ?? "No file chosen"}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 pt-2">
        <button
          type="button"
          className="rounded bg-[var(--accent)] px-4 py-2 text-xs font-semibold text-white hover:opacity-90 transition-opacity disabled:cursor-not-allowed disabled:opacity-40 shadow-sm"
          data-testid="upload-analyze"
          disabled={!allReady || busy}
          onClick={onAnalyze}
        >
          {busy ? "Analyzing uploaded Jan/Feb/Mar data..." : "Analyze Uploaded Data"}
        </button>

        {!allReady && !busy && (
          <span className="text-xs text-[var(--text-muted)]">
            Select test data archives for all 3 months to proceed.
          </span>
        )}
      </div>

      {busy ? (
        <div
          className="mt-4 space-y-2 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-panel-secondary)] p-3.5"
          data-testid="upload-progress"
        >
          <div className="flex items-center justify-between text-xs font-mono text-[var(--accent)]">
            <span className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-[var(--accent)] animate-pulse" />
              {jobStage || "Analyzing uploaded Jan/Feb/Mar data..."}
            </span>
            <span className="font-bold">{progressPct ?? 5}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--border-subtle)]">
            <div
              className="h-full bg-[var(--accent)] transition-all duration-300 rounded-full"
              style={{ width: `${Math.min(100, Math.max(5, progressPct ?? 5))}%` }}
            />
          </div>
        </div>
      ) : null}

      {error ? (
        <p className="mt-3 text-xs font-medium text-red-500" data-testid="upload-analysis-error">
          {error}
        </p>
      ) : null}
    </section>
  );
}

/** @deprecated Prefer UploadAnalysisPanel for production three-month workflow. */
export { UploadAnalysisPanel as UploadTestDataPanel };
