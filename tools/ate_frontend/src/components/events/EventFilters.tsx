"use client";

import type { EventFilterOptions, EventFiltersState, EventSeverity } from "@/types/events";
import { EMPTY_EVENT_FILTERS } from "@/types/events";

const ALL_SEVERITIES: EventSeverity[] = ["CRITICAL", "ERROR", "WARN", "PASS", "INFO"];

interface Props {
  filters: EventFiltersState;
  options?: EventFilterOptions;
  onChange: (next: EventFiltersState) => void;
}

export function EventFilters({ filters, options, onChange }: Props) {
  const set = <K extends keyof EventFiltersState>(key: K, value: EventFiltersState[K]) => {
    onChange({ ...filters, [key]: value });
  };

  const toggleSeverity = (sev: EventSeverity) => {
    const has = filters.severity.includes(sev);
    set(
      "severity",
      has ? filters.severity.filter((s) => s !== sev) : [...filters.severity, sev],
    );
  };

  return (
    <div className="flex flex-col gap-2.5 border-b border-[var(--line)] pb-3">
      <div className="grid grid-cols-1 gap-2 md:grid-cols-[1.4fr_repeat(4,1fr)]">
        <label className="flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          Search
          <input
            value={filters.q}
            onChange={(e) => set("q", e.target.value)}
            placeholder="message, lot, die, type…"
            className="vl-field px-2.5 py-1.5 text-[11.5px] normal-case tracking-normal"
          />
        </label>
        <FilterSelect
          label="Tester"
          value={filters.tester_id}
          options={options?.testers ?? []}
          onChange={(v) => set("tester_id", v)}
        />
        <FilterSelect
          label="Site"
          value={filters.site_id}
          options={options?.sites ?? []}
          onChange={(v) => set("site_id", v)}
        />
        <FilterSelect
          label="Lot"
          value={filters.lot_id}
          options={options?.lots ?? []}
          onChange={(v) => set("lot_id", v)}
        />
        <FilterSelect
          label="Wafer"
          value={filters.wafer_id}
          options={options?.wafers ?? []}
          onChange={(v) => set("wafer_id", v)}
        />
      </div>

      <div className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_1fr_1fr_auto]">
        <label className="flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          Since
          <input
            type="datetime-local"
            value={filters.since}
            onChange={(e) => set("since", e.target.value)}
            className="vl-field px-2.5 py-1.5 font-mono text-[11px] normal-case tracking-normal"
          />
        </label>
        <label className="flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          Until
          <input
            type="datetime-local"
            value={filters.until}
            onChange={(e) => set("until", e.target.value)}
            className="vl-field px-2.5 py-1.5 font-mono text-[11px] normal-case tracking-normal"
          />
        </label>
        <label className="flex flex-col gap-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          Ack state
          <select
            value={filters.acknowledged}
            onChange={(e) => set("acknowledged", e.target.value as EventFiltersState["acknowledged"])}
            className="vl-field px-2.5 py-1.5 text-[11.5px] normal-case tracking-normal"
          >
            <option value="">All</option>
            <option value="false">Unacknowledged</option>
            <option value="true">Acknowledged</option>
          </select>
        </label>
        <div className="flex items-end">
          <button
            type="button"
            onClick={() => onChange({ ...EMPTY_EVENT_FILTERS })}
            className="vl-field rounded-[var(--radius)] px-3 py-1.5 text-[11px] text-[var(--muted)] transition-[border-color,color,background-color] duration-200 hover:border-[var(--line-bright)] hover:text-[var(--text)]"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[10px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
          Severity
        </span>
        {ALL_SEVERITIES.map((sev) => {
          const on = filters.severity.includes(sev);
          return (
            <button
              key={sev}
              type="button"
              onClick={() => toggleSeverity(sev)}
              className={`rounded-[var(--radius)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.04em] transition-[border-color,background-color,color] duration-200 ${
                on
                  ? "border border-[rgba(107,193,242,0.35)] bg-[var(--cyan-dim)] text-[var(--cyan)]"
                  : "border border-[var(--line)] text-[var(--muted-2)] hover:border-[var(--line-bright)] hover:text-[var(--muted)]"
              }`}
            >
              {sev}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function FilterSelect({
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
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}
