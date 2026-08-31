import { Combobox } from "@headlessui/react";
import { ChevronUpDownIcon } from "@heroicons/react/24/solid";
import { useMemo, useState } from "react";

interface SearchableSelectProps {
  label: string;
  value: string;
  options: string[];
  placeholder: string;
  disabled?: boolean;
  loading?: boolean;
  emptyMessage?: string;
  loadingMessage?: string;
  onChange: (value: string) => void;
}

export function SearchableSelect({
  label,
  value,
  options,
  placeholder,
  disabled,
  loading,
  emptyMessage,
  loadingMessage,
  onChange,
}: SearchableSelectProps) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) => o.toLowerCase().includes(q));
  }, [options, query]);

  return (
    <label className="block text-xs text-gray-400">
      {label}
      <Combobox
        value={value}
        onChange={(v: string | null) => onChange(v ?? "")}
        disabled={disabled}
      >
        <div className="relative mt-1">
          <Combobox.Input
            className="w-full rounded border border-gray-700 bg-gray-950 px-3 py-2 pr-9 text-sm font-mono text-gray-100 disabled:opacity-50"
            onChange={(event) => setQuery(event.target.value)}
            displayValue={(v: string) => v}
            placeholder={loading ? (loadingMessage ?? "Loading...") : placeholder}
          />
          <Combobox.Button className="absolute inset-y-0 right-0 flex items-center px-2 text-gray-500">
            <ChevronUpDownIcon className="h-4 w-4" aria-hidden="true" />
          </Combobox.Button>
          {!disabled && (
            <Combobox.Options className="absolute z-20 mt-1 max-h-56 w-full overflow-auto rounded border border-gray-700 bg-gray-900 py-1 text-sm shadow-lg">
              {loading ? (
                <li className="px-3 py-2 text-gray-400">{loadingMessage ?? "Loading..."}</li>
              ) : filtered.length === 0 ? (
                <li className="px-3 py-2 text-gray-500">{emptyMessage ?? "No options available"}</li>
              ) : (
                filtered.map((item) => (
                  <Combobox.Option
                    key={item}
                    value={item}
                    className={({ active }) =>
                      `cursor-pointer px-3 py-2 ${
                        active ? "bg-cyan-900 text-cyan-100" : "text-gray-200"
                      }`
                    }
                  >
                    {item}
                  </Combobox.Option>
                ))
              )}
            </Combobox.Options>
          )}
        </div>
      </Combobox>
    </label>
  );
}
