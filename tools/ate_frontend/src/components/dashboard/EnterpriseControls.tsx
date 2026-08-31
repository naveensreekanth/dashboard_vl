"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { fetchEventFilterOptions } from "@/services/api";
import { useOpsStore } from "@/stores/opsStore";

/**
 * Enterprise floor controls — selection + live ops.
 * Values drive wafer focus, event log scope, and stream pause/reconnect.
 * Never invents metrics; only scopes authoritative backend data.
 */
export function EnterpriseControls() {
  const queryClient = useQueryClient();
  const lotId = useOpsStore((s) => s.lotId);
  const waferId = useOpsStore((s) => s.waferId);
  const testerId = useOpsStore((s) => s.testerId);
  const siteId = useOpsStore((s) => s.siteId);
  const since = useOpsStore((s) => s.since);
  const until = useOpsStore((s) => s.until);
  const streamMode = useOpsStore((s) => s.streamMode);
  const setLotId = useOpsStore((s) => s.setLotId);
  const setWaferId = useOpsStore((s) => s.setWaferId);
  const setTesterId = useOpsStore((s) => s.setTesterId);
  const setSiteId = useOpsStore((s) => s.setSiteId);
  const setSince = useOpsStore((s) => s.setSince);
  const setUntil = useOpsStore((s) => s.setUntil);
  const toggleStreamMode = useOpsStore((s) => s.toggleStreamMode);
  const requestReconnect = useOpsStore((s) => s.requestReconnect);

  const { data: options } = useQuery({
    queryKey: ["event-filter-options"],
    queryFn: fetchEventFilterOptions,
    staleTime: 60_000,
  });

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
    void queryClient.invalidateQueries({ queryKey: ["kpis"] });
    void queryClient.invalidateQueries({ queryKey: ["test-events"] });
    void queryClient.invalidateQueries({ queryKey: ["maintenance"] });
    void queryClient.invalidateQueries({ queryKey: ["test-limits"] });
    void queryClient.invalidateQueries({ queryKey: ["wafer"] });
  };

  return (
    <section className="vl-surface mb-5 p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="vl-label">Enterprise Floor Controls</div>
        <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-[var(--muted)]">
          Stream{" "}
          <span
            className={
              streamMode === "LIVE" ? "text-[var(--green)]" : "text-[var(--amber)]"
            }
          >
            {streamMode}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-6">
        <SelectField
          label="Lot"
          value={lotId}
          options={options?.lots ?? []}
          onChange={setLotId}
        />
        <SelectField
          label="Wafer"
          value={waferId}
          options={options?.wafers ?? []}
          onChange={setWaferId}
        />
        <SelectField
          label="Tester"
          value={testerId}
          options={options?.testers ?? []}
          onChange={setTesterId}
        />
        <SelectField
          label="Site"
          value={siteId}
          options={options?.sites ?? []}
          onChange={setSiteId}
        />
        <label className="flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          Since
          <input
            type="datetime-local"
            value={since}
            onChange={(e) => setSince(e.target.value)}
            className="vl-field px-2.5 py-1.5 font-mono text-[11px] normal-case tracking-normal"
          />
        </label>
        <label className="flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          Until
          <input
            type="datetime-local"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
            className="vl-field px-2.5 py-1.5 font-mono text-[11px] normal-case tracking-normal"
          />
        </label>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <Button
          type="button"
          variant={streamMode === "LIVE" ? "default" : "ghost"}
          onClick={toggleStreamMode}
        >
          {streamMode === "LIVE" ? "Pause live" : "Resume live"}
        </Button>
        <Button type="button" onClick={refresh}>
          Refresh
        </Button>
        <Button
          type="button"
          onClick={() => {
            requestReconnect();
            refresh();
          }}
        >
          Reconnect
        </Button>
      </div>
    </section>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="vl-field px-2.5 py-1.5 text-[11.5px] normal-case tracking-normal"
      >
        <option value="">All</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}
