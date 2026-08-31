interface ErrorStateProps {
  message: string;
  code?: string | null;
}

export function ErrorState({ message, code }: ErrorStateProps) {
  return (
    <div
      className="rounded-lg border border-red-500/40 bg-red-500/5 dark:bg-red-950/20 p-4 shadow-sm"
      role="alert"
      aria-live="assertive"
    >
      <p className="text-xs font-bold uppercase tracking-wider text-red-600 dark:text-red-400">Recommendation service error</p>
      <p className="mt-1 text-xs text-[var(--text-primary)]">{message}</p>
      {code && <p className="mt-2 text-[10px] font-mono text-[var(--text-muted)]">Code: {code}</p>}
    </div>
  );
}
