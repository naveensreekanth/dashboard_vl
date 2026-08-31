"use client";

import { useState, type MouseEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { DetailPopup } from "@/components/common/DetailPopup";
import type { LimitStatus, TestLimitOut, TestLimitsOut } from "@/types/api";
import {
  approveTestLimit,
  rejectTestLimit,
  rollbackTestLimit,
} from "@/services/api";
import { formatTime } from "@/lib/utils";

const STATUS_STYLE: Record<LimitStatus, string> = {
  ACTIVE: "text-[var(--green)]",
  PENDING_APPROVAL: "text-[var(--amber)]",
  RECOMMENDED: "text-[var(--cyan)]",
  REJECTED: "text-[var(--muted)]",
  ROLLED_BACK: "text-[var(--muted-2)]",
};

export function DynamicTestLimits({ data }: { data: TestLimitsOut | null }) {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<TestLimitOut | null>(null);

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["dashboard"] });
    void queryClient.invalidateQueries({ queryKey: ["test-limits"] });
  };

  const approveMut = useMutation({
    mutationFn: (limitId: string) => approveTestLimit(limitId, { actor: "engineer" }),
    onSuccess: invalidate,
  });
  const rejectMut = useMutation({
    mutationFn: (limitId: string) => rejectTestLimit(limitId, { actor: "engineer" }),
    onSuccess: invalidate,
  });
  const rollbackMut = useMutation({
    mutationFn: (limitId: string) => rollbackTestLimit(limitId, { actor: "engineer" }),
    onSuccess: invalidate,
  });

  const busy = approveMut.isPending || rejectMut.isPending || rollbackMut.isPending;

  return (
    <>
      <div className="vl-card flex flex-col gap-2.5 overflow-hidden p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => window.dispatchEvent(new CustomEvent("switchSuiteTab", { detail: { tab: "dtl" } }))}
              className="rounded border border-[rgba(107,193,242,0.3)] bg-[rgba(107,193,242,0.12)] px-2 py-0.5 text-[10px] font-bold text-[var(--cyan)] hover:bg-[rgba(107,193,242,0.22)]"
            >
              Open DTL Tool ↗
            </button>
            <div className="text-[13px] font-semibold text-[#f2f7fc]">Dynamic Test Limits</div>
          </div>
          <span className="rounded bg-[var(--cyan-dim)] px-[7px] py-0.5 text-[10px] font-semibold text-[var(--cyan)]">
            {data ? `${data.adjustments_today} today` : "— today"}
          </span>
        </div>
        <div>
          {(data?.items ?? []).map((item) => (
            <LimitRow
              key={item.limit_id}
              item={item}
              busy={busy}
              onOpen={() => setSelected(item)}
              onApprove={() => approveMut.mutate(item.limit_id)}
              onReject={() => rejectMut.mutate(item.limit_id)}
              onRollback={() => rollbackMut.mutate(item.limit_id)}
            />
          ))}
          {!data?.items?.length ? (
            <div className="border-t border-[var(--line)] py-[7px] text-[11.5px] text-[var(--muted)] first:border-t-0">
              No limit adjustments from backend
            </div>
          ) : null}
        </div>
        <div className="text-[11.5px] leading-relaxed text-[var(--muted)]">
          Per-lot limit tightening driven by rolling process-capability (Cpk) trends.
          Approvals apply authoritative backend state only.
        </div>
      </div>

      {selected ? (
        <LimitDetailPopup
          item={
            data?.items.find((i) => i.limit_id === selected.limit_id) ?? selected
          }
          busy={busy}
          onClose={() => setSelected(null)}
          onApprove={() => approveMut.mutate(selected.limit_id)}
          onReject={() => rejectMut.mutate(selected.limit_id)}
          onRollback={() => rollbackMut.mutate(selected.limit_id)}
        />
      ) : null}
    </>
  );
}

function LimitRow({
  item,
  busy,
  onOpen,
  onApprove,
  onReject,
  onRollback,
}: {
  item: TestLimitOut;
  busy: boolean;
  onOpen: () => void;
  onApprove: () => void;
  onReject: () => void;
  onRollback: () => void;
}) {
  const pending = item.status === "PENDING_APPROVAL" || item.status === "RECOMMENDED";
  const canRollback = item.status === "ACTIVE" || item.status === "REJECTED";
  const name = item.name || `${item.parameter} · ${item.test_name}`;

  return (
    <div className="border-t border-[var(--line)] py-[7px] text-[11.5px] first:border-t-0">
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full items-center justify-between gap-2 rounded-[4px] px-1 text-left transition-[color,background-color] duration-200 hover:bg-[rgba(107,193,242,0.06)] hover:text-[var(--cyan)]"
      >
        <div className="min-w-0">
          <div className="truncate text-[var(--text)]">{name}</div>
          <div className={`mt-0.5 font-mono text-[10px] ${STATUS_STYLE[item.status]}`}>
            {item.status}
            {item.cpk != null ? ` · Cpk ${item.cpk.toFixed(2)}` : ""}
          </div>
        </div>
        <div className="shrink-0 text-right font-mono text-[11px] font-semibold text-[var(--cyan)]">
          {item.change_label}
        </div>
      </button>
      {pending || canRollback ? (
        <div className="mt-1.5 flex gap-1.5">
          {pending ? (
            <>
              <ActionBtn
                label="Approve"
                disabled={busy}
                onClick={(e) => {
                  e.stopPropagation();
                  onApprove();
                }}
              />
              <ActionBtn
                label="Reject"
                disabled={busy}
                muted
                onClick={(e) => {
                  e.stopPropagation();
                  onReject();
                }}
              />
            </>
          ) : null}
          {canRollback ? (
            <ActionBtn
              label="Rollback"
              disabled={busy}
              muted
              onClick={(e) => {
                e.stopPropagation();
                onRollback();
              }}
            />
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function LimitDetailPopup({
  item,
  busy,
  onClose,
  onApprove,
  onReject,
  onRollback,
}: {
  item: TestLimitOut;
  busy: boolean;
  onClose: () => void;
  onApprove: () => void;
  onReject: () => void;
  onRollback: () => void;
}) {
  const pending = item.status === "PENDING_APPROVAL" || item.status === "RECOMMENDED";
  const canRollback = item.status === "ACTIVE" || item.status === "REJECTED";
  const name = item.name || `${item.parameter} · ${item.test_name}`;

  return (
    <DetailPopup eyebrow="Dynamic Test Limits" title={name} onClose={onClose} wide>
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
        <Tile label="Status" value={item.status} accentClass={STATUS_STYLE[item.status]} />
        <Tile label="Change" value={item.change_label} />
        <Tile label="Direction" value={item.direction} />
        <Tile label="Previous" value={String(item.previous_limit)} />
        <Tile label="Current" value={String(item.current_limit)} />
        <Tile label="Delta" value={String(item.delta)} />
        <Tile label="Cpk" value={item.cpk != null ? item.cpk.toFixed(2) : "—"} />
        <Tile label="Target Cpk" value={String(item.target_cpk)} />
        <Tile
          label="Confidence"
          value={item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : "—"}
        />
      </div>

      <div className="mb-4 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-3">
        <Meta label="Parameter" value={item.parameter} />
        <Meta label="Test" value={item.test_name} />
        <Meta label="Lot" value={item.lot_id ?? "—"} />
        <Meta label="Tester" value={item.tester_id ?? "—"} />
        <Meta label="Site" value={item.site_id ?? "—"} />
        <Meta label="Limit ID" value={item.limit_id} />
      </div>

      {item.reason ? (
        <p className="mb-4 text-[12px] leading-relaxed text-[var(--muted)]">{item.reason}</p>
      ) : null}

      <div className="mb-4 font-mono text-[10px] text-[var(--muted-2)]">
        Created {formatTime(item.created_at)} · Updated {formatTime(item.updated_at)}
      </div>

      {pending || canRollback ? (
        <div className="flex flex-wrap gap-2 border-t border-[var(--line)] pt-3">
          {pending ? (
            <>
              <ActionBtn label="Approve" disabled={busy} onClick={() => onApprove()} />
              <ActionBtn label="Reject" disabled={busy} muted onClick={() => onReject()} />
            </>
          ) : null}
          {canRollback ? (
            <ActionBtn label="Rollback" disabled={busy} muted onClick={() => onRollback()} />
          ) : null}
        </div>
      ) : null}
    </DetailPopup>
  );
}

function Tile({
  label,
  value,
  accentClass,
}: {
  label: string;
  value: string;
  accentClass?: string;
}) {
  return (
    <div className="vl-popup-tile px-3 py-2.5">
      <div className="vl-popup-tile-label">{label}</div>
      <div
        className={`vl-popup-tile-value mt-1 font-mono text-[14px] font-semibold ${accentClass ?? ""}`}
      >
        {value}
      </div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="vl-popup-tile px-3 py-2">
      <div className="vl-popup-tile-label">{label}</div>
      <div className="vl-popup-tile-value mt-0.5 font-mono text-[12px] font-semibold">{value}</div>
    </div>
  );
}

function ActionBtn({
  label,
  onClick,
  disabled,
  muted,
}: {
  label: string;
  onClick: (e: MouseEvent) => void;
  disabled?: boolean;
  muted?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`rounded border px-2 py-0.5 text-[10px] font-semibold disabled:opacity-40 ${
        muted
          ? "border-[var(--line)] text-[var(--muted)]"
          : "border-[var(--cyan)] text-[var(--cyan)]"
      }`}
    >
      {label}
    </button>
  );
}
