import { Disclosure, DisclosureButton, DisclosurePanel } from "@headlessui/react";
import type { AuditRecord, LotRecommendationResult } from "@/api/types";

interface AuditPanelProps {
  result: LotRecommendationResult;
}

export function AuditPanel({ result }: AuditPanelProps) {
  const audit: AuditRecord = result.audit;

  return (
    <Disclosure as="section" className="rounded-lg border border-gray-800 bg-gray-900">
      {({ open }) => (
        <>
          <DisclosureButton className="flex w-full items-center justify-between p-4 text-left">
            <h2 className="text-sm font-semibold text-gray-200">Audit / Provenance</h2>
            <span className="text-gray-500 text-xs">{open ? "Hide" : "Expand"}</span>
          </DisclosureButton>
          <DisclosurePanel className="border-t border-gray-800 p-4">
            <dl className="grid grid-cols-2 gap-3 text-xs font-mono text-gray-300 mb-4">
              <div>
                <dt className="text-gray-500">request_id</dt>
                <dd className="break-all">{audit.request_id ?? result.request_id}</dd>
              </div>
              <div>
                <dt className="text-gray-500">timestamp</dt>
                <dd>{audit.timestamp ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-gray-500">model_version</dt>
                <dd>{audit.model_version ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-gray-500">dataset_version</dt>
                <dd>{audit.dataset_version ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-gray-500">ml_dataset_version</dt>
                <dd>{audit.ml_dataset_version ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-gray-500">policy_config_version</dt>
                <dd>{audit.policy_config_version ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-gray-500">TOP_N</dt>
                <dd>{audit.TOP_N ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-gray-500">joint_enabled</dt>
                <dd>{String(audit.joint_enabled ?? false)}</dd>
              </div>
              <div>
                <dt className="text-gray-500">evidence_origin</dt>
                <dd>{audit.evidence_origin ?? "—"}</dd>
              </div>
            </dl>
            <details className="text-xs">
              <summary className="cursor-pointer text-gray-400 mb-2">Full audit JSON</summary>
              <pre className="overflow-auto rounded bg-gray-950 p-3 text-gray-400 max-h-96">
                {JSON.stringify(audit, null, 2)}
              </pre>
            </details>
          </DisclosurePanel>
        </>
      )}
    </Disclosure>
  );
}
