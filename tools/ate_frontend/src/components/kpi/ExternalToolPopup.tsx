"use client";

import type { ReactNode } from "react";
import { DetailPopup } from "@/components/common/DetailPopup";

/**
 * Shared popup for external KPI tools (Retest Streamlit, SHMOO VL, etc.).
 */
export function ExternalToolPopup({
  title,
  onClose,
  description,
  ctaLabel,
  ctaHref,
  children,
}: {
  title: string;
  onClose: () => void;
  description: string;
  ctaLabel: string;
  ctaHref: string;
  children?: ReactNode;
}) {
  return (
    <DetailPopup eyebrow="External tool" title={title} onClose={onClose} wide>
      <div className="flex flex-col gap-3">
        <div className="vl-external-cta">
          <p className="vl-external-cta-text">{description}</p>
          <a
            href={ctaHref}
            target="_blank"
            rel="noopener noreferrer"
            className="vl-btn-primary"
          >
            {ctaLabel}
          </a>
        </div>
        {children ? <div className="min-w-0">{children}</div> : null}
      </div>
    </DetailPopup>
  );
}
