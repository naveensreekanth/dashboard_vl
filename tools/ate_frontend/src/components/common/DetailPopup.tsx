"use client";

import { useEffect, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

/**
 * Centered detail popup portaled to body.
 * Esc / backdrop / Close dismisses.
 */
export function DetailPopup({
  title,
  eyebrow,
  onClose,
  children,
  wide,
}: {
  title: string;
  eyebrow?: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!mounted) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[80] flex items-start justify-center p-4 pt-[8vh] sm:p-6 sm:pt-[10vh]"
      role="presentation"
    >
      <button
        type="button"
        className="vl-popup-backdrop fixed inset-0 bg-black/50 transition-opacity duration-[240ms]"
        aria-label="Close dialog"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`vl-popup vl-enter relative flex max-h-[min(90vh,860px)] w-full flex-col overflow-hidden ${
          wide ? "max-w-[1100px]" : "max-w-[560px]"
        }`}
      >
        <header className="flex shrink-0 items-start justify-between border-b border-[rgba(107,193,242,0.28)] px-5 py-4">
          <div>
            {eyebrow ? <div className="vl-popup-label">{eyebrow}</div> : null}
            <h2 className="font-display mt-1 text-[22px] font-semibold text-[var(--text-bright)]">
              {title}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-[6px] border border-[rgba(107,193,242,0.45)] bg-[var(--cyan-dim)] px-2.5 py-1 text-[11px] font-semibold text-[var(--cyan)] transition-[border-color,color,background-color] duration-200 hover:border-[var(--cyan)] hover:text-[var(--text-bright)]"
          >
            Close
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
      </div>
    </div>,
    document.body,
  );
}
