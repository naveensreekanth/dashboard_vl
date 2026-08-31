export function LoadingState({ label = "Loading floor telemetry…" }: { label?: string }) {
  return <div className="vl-state-panel">{label}</div>;
}
