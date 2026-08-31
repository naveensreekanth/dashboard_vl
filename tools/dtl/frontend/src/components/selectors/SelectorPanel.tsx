import { useSelectorData } from "@/hooks/useSelectorData";
import { useAppContext } from "@/state/useAppContext";
import { useRecommendation } from "@/hooks/useRecommendation";
import { SearchableSelect } from "./SearchableSelect";

const PRODUCTION_MONTH_OPTIONS = [
  { value: "", label: "Legacy (no month)" },
  { value: "2026-01", label: "2026-01" },
  { value: "2026-02", label: "2026-02" },
  { value: "2026-03", label: "2026-03" },
];

export function SelectorPanel() {
  const { state, dispatch } = useAppContext();
  const { fetchRecommendation } = useRecommendation();
  useSelectorData();

  return (
    <section
      className="rounded-lg border border-gray-800 bg-gray-900 p-4"
      aria-label="Lot die parameter selector"
    >
      <h2 className="text-sm font-semibold text-gray-200 mb-4">Analysis Context</h2>
      <div className="grid gap-4 md:grid-cols-5">
        <SearchableSelect
          label="Lot"
          value={state.selectedLot}
          options={state.availableLots}
          onChange={(value) => dispatch({ type: "SET_LOT", payload: value })}
          placeholder="Select Lot"
          loading={state.selectorsLoading.lots}
          loadingMessage="Loading lots..."
          emptyMessage="No lots available"
        />
        <SearchableSelect
          label="Die"
          value={state.selectedDie}
          options={state.availableDies}
          onChange={(value) => dispatch({ type: "SET_DIE", payload: value })}
          placeholder={state.selectedLot ? "Select Die" : "Select Lot first"}
          disabled={!state.selectedLot}
          loading={state.selectorsLoading.dies}
          loadingMessage="Loading dies..."
          emptyMessage="No dies available for this lot"
        />
        <label className="block text-xs text-gray-400">
          Parameter
          <select
            value={state.selectedParameter}
            onChange={(e) => dispatch({ type: "SET_PARAMETER", payload: e.target.value })}
            disabled={!state.selectedDie || state.selectorsLoading.parameters}
            className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 disabled:opacity-50"
          >
            <option value="" disabled>
              {state.selectorsLoading.parameters
                ? "Loading parameters..."
                : state.selectedDie
                  ? "Select Parameter"
                  : "Select Die first"}
            </option>
            {state.availableParameters.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs text-gray-400">
          Production month
          <select
            value={state.selectedProductionMonth}
            onChange={(e) =>
              dispatch({ type: "SET_PRODUCTION_MONTH", payload: e.target.value })
            }
            className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100"
            data-testid="production-month-select"
          >
            {PRODUCTION_MONTH_OPTIONS.map((o) => (
              <option key={o.value || "legacy"} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <div className="flex items-end">
          <button
            type="button"
            onClick={() => void fetchRecommendation()}
            disabled={
              state.loading ||
              state.serviceReady === false ||
              !state.selectedLot ||
              !state.selectedDie ||
              !state.selectedParameter
            }
            className="w-full rounded bg-cyan-700 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {state.loading ? "Loading recommendation…" : "Run Recommendation"}
          </button>
        </div>
      </div>
    </section>
  );
}
