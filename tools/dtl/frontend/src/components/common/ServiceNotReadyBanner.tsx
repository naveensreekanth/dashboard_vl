export function ServiceNotReadyBanner() {
  return (
    <div className="rounded-lg border border-amber-800 bg-amber-950/30 p-4" role="status">
      <p className="text-sm font-medium text-amber-400">DTL recommendation service is warming up</p>
      <p className="mt-1 text-xs text-gray-400">
        The service is not ready. Recommendations cannot be processed until readiness checks pass.
      </p>
    </div>
  );
}
