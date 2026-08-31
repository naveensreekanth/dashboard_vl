export function LoadingState({ message = "Loading recommendation…" }: { message?: string }) {
  return (
    <div
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] p-6 text-center shadow-sm"
      role="status"
      aria-live="polite"
    >
      <div className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-[var(--accent)] border-t-transparent" />
      <p className="mt-3 text-xs text-[var(--text-muted)] font-mono">{message}</p>
    </div>
  );
}
