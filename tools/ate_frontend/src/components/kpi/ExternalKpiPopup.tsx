"use client";

import { useState } from "react";
import { DetailPopup } from "@/components/common/DetailPopup";
import { ExternalToolPopup } from "@/components/kpi/ExternalToolPopup";
import { RetestLiveFeed } from "@/components/kpi/RetestLiveFeed";
import { KpiMetricTile } from "@/components/kpi/KpiMetricTile";
import { RetestFlashIcon } from "@/components/kpi/RetestFlashIcon";
import { TestTimePreview } from "@/components/kpi/TestTimePreview";
import {
  isExternalToolOnlyUrl,
  isPlaceholderKpiUrl,
  isStreamlitKpiUrl,
  RETEST_AI_RECOMMENDATIONS,
  type TestTimeCapabilityMetric,
} from "@/lib/kpiExternalPages";

/**
 * Centered popup that embeds a separately deployed KPI page.
 * Streamlit / Render tools open externally — they are unreliable in iframes.
 */
export function ExternalKpiPopup({
  title,
  url,
  onClose,
  kpiId,
  testTimeMetrics,
}: {
  title: string;
  url: string;
  onClose: () => void;
  kpiId?: string;
  testTimeMetrics?: TestTimeCapabilityMetric[];
}) {
  const [loaded, setLoaded] = useState(false);
  const placeholder = isPlaceholderKpiUrl(url);
  const externalOnly = isExternalToolOnlyUrl(url);
  const streamlit = isStreamlitKpiUrl(url);
  const showRetestSummary = kpiId === "retest_reduction";
  const showTestTimeSummary = kpiId === "test_time_reduction";

  if (externalOnly) {
    const isTestTime = kpiId === "test_time_reduction";
    return (
      <ExternalToolPopup
        title={title}
        onClose={onClose}
        description={
          streamlit
            ? "The Retest AI Agent runs on Streamlit Cloud and cannot be embedded here (browser blocks the redirect loop). Open it in a new tab to analyze uploads and recommendations."
            : isTestTime
              ? "The Test Time Optimization agent runs on a separate page. Open it in a new tab to run simulations for test time and vector memory optimization."
              : "This tool runs on a separate page and cannot be embedded here. Open it in a new tab to continue."
        }
        ctaLabel={
          streamlit
            ? "Open Retest AI Agent"
            : isTestTime
              ? "Open Test Time Optimization"
              : "Open external tool"
        }
        ctaHref={url}
      >
        {showRetestSummary ? (
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-1 gap-x-6 gap-y-3 border-t border-[rgba(107,193,242,0.14)] pt-3 sm:grid-cols-2">
              {RETEST_AI_RECOMMENDATIONS.map((rec) => (
                <KpiMetricTile
                  key={rec.id}
                  icon={<RetestFlashIcon size={12} />}
                  eyebrow="AI Recommended"
                  title={rec.title}
                  size="popup"
                  metrics={[
                    { label: "Events", value: rec.events, color: rec.valueColor },
                    { label: "Devices", value: rec.devices, color: rec.valueColor },
                  ]}
                />
              ))}
            </div>
            <RetestLiveFeed />
          </div>
        ) : null}
        {showTestTimeSummary ? (
          <div className="flex flex-col gap-3 border-t border-[rgba(107,193,242,0.14)] pt-3">
            {(testTimeMetrics?.length ?? 0) > 0 ? (
              <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
                {testTimeMetrics!.map((m) => (
                  <KpiMetricTile
                    key={m.id}
                    eyebrow="Capability"
                    title={m.label}
                    size="popup"
                    metrics={[
                      {
                        label: m.primaryLabel ?? "Value",
                        value: Number.isFinite(m.value) ? m.value : "—",
                        unit: Number.isFinite(m.value) ? m.unit : undefined,
                      },
                    ]}
                  />
                ))}
              </div>
            ) : null}
            <TestTimePreview />
          </div>
        ) : null}
      </ExternalToolPopup>
    );
  }

  return (
    <DetailPopup eyebrow="External tool" title={title} onClose={onClose} wide>
      {placeholder ? (
        <div className="mb-3 rounded border border-[var(--amber)]/40 bg-[var(--amber-dim)] px-3 py-2 text-[11px] leading-relaxed text-[var(--amber)]">
          Placeholder URL — replace with your real Vercel deploy via{" "}
          <span className="font-mono">NEXT_PUBLIC_KPI_*</span> env vars.
        </div>
      ) : null}

      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[11px] font-semibold text-[var(--cyan)] underline-offset-2 hover:underline"
        >
          Open in new tab
        </a>
        {!loaded && !placeholder ? (
          <span className="text-[10px] text-[var(--muted-2)]">Loading page…</span>
        ) : null}
      </div>

      <div className="overflow-hidden rounded-[var(--radius-card)] border border-[rgba(107,193,242,0.25)] bg-[var(--popup-panel)]">
        <iframe
          title={title}
          src={url}
          className="h-[min(70vh,720px)] w-full border-0"
          onLoad={() => setLoaded(true)}
          allow="fullscreen"
          referrerPolicy="no-referrer-when-downgrade"
        />
      </div>
    </DetailPopup>
  );
}
