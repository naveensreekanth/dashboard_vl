export function AboutPage() {
  return (
    <div className="space-y-6 max-w-3xl text-sm text-gray-300">
      <div>
        <h1 className="text-xl font-semibold text-gray-100">About</h1>
        <p className="mt-2 text-gray-400">
          DTL AI Agent Engineering Dashboard — Phase 10 presentation layer.
        </p>
      </div>
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4 space-y-2">
        <h2 className="font-semibold text-gray-200">Decision support only</h2>
        <p>
          This dashboard is <strong>decision support</strong>, not automatic DTL limit deployment.
          All recommendations, safety evaluations, and policy decisions are produced by the Phase 8
          backend engine.
        </p>
      </section>
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4 space-y-2">
        <h2 className="font-semibold text-gray-200">Evidence limitations</h2>
        <ul className="list-disc pl-5 text-gray-400 space-y-1">
          <li>Simulation evidence is SIMULATOR_DERIVED.</li>
          <li>objective_score is not production reliability or true optimality.</li>
          <li>SYNTHETIC_ASSUMED limits are shown when applicable.</li>
          <li>Joint model is DISABLED.</li>
          <li>Tree baseline is DIAGNOSTIC / REFERENCE ONLY when enabled.</li>
        </ul>
      </section>
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4 space-y-2">
        <h2 className="font-semibold text-gray-200">API</h2>
        <p className="text-gray-400">
          Consumes Phase 9 FastAPI at <code className="text-gray-500">/api/v1</code>. No CSV access,
          no client-side ranking, no safety re-evaluation.
        </p>
      </section>
    </div>
  );
}
