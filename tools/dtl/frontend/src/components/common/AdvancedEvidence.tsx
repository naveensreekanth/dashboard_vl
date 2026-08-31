import { Disclosure, DisclosureButton } from "@headlessui/react";
import type { ReactNode } from "react";

interface AdvancedEvidenceProps {
  children: ReactNode;
}

/** Collapsible container for engineering investigation panels. Closed content is unmounted. */
export function AdvancedEvidence({ children }: AdvancedEvidenceProps) {
  return (
    <Disclosure
      as="section"
      defaultOpen={false}
      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-panel)] shadow-sm"
      data-testid="advanced-evidence"
    >
      {({ open }) => (
        <>
          <DisclosureButton className="flex w-full items-center justify-between p-4 text-left hover:bg-[var(--bg-panel-secondary)] transition-colors">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)]">Advanced Evidence</h2>
              <p className="mt-0.5 text-xs text-[var(--text-muted)]">
                Simulation evidence, safety gate, model and audit provenance.
              </p>
            </div>
            <span className="text-[var(--text-muted)] text-xs font-mono shrink-0 ml-4">
              {open ? "Hide ▲" : "Show ▼"}
            </span>
          </DisclosureButton>
          {open ? (
            <div
              className="border-t border-[var(--border-subtle)] p-4 space-y-6"
              data-testid="advanced-evidence-body"
            >
              {children}
            </div>
          ) : null}
        </>
      )}
    </Disclosure>
  );
}
