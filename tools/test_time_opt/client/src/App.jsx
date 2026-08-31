import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { recommendNextChipGreedyMl } from "./nextChipRecommend";

const API = "";
const FIRST_N = 10;
const THEME_KEY = "verilumen-theme";

function readStoredTheme() {
  try {
    const t = localStorage.getItem(THEME_KEY);
    if (t === "light" || t === "dark") return t;
  } catch {
    /* ignore */
  }
  if (typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }
  return "dark";
}

/** Advantest logs use P1..P1000; Verilumen / STIL ids are 0..999. */
function logLabelToPatternId(label) {
  return label <= 0 ? label : label - 1;
}

/** One die/chip id per log: folder name if nested, else file basename. */
function dieKeyFromRel(rel) {
  const parts = String(rel || "")
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean);
  if (parts.length >= 2) return parts[0];
  const name = parts[0] || "(root)";
  return name.replace(/\.[^.]+$/, "") || name;
}

/** @deprecated use dieKeyFromRel — kept name alias for call sites */
function folderKeyFromRel(rel) {
  return dieKeyFromRel(rel);
}

/**
 * Parse one Advantest-style die log →
 * { [patternId]: { fails, passes, channel_fails, fail_channels } }.
 */
function parseAteLogText(text) {
  const atePat = /^P(\d+)\s*\|\s*(?:CH(\d+))?/i;
  const ateStatus = /STATUS\s*:\s*([FP])\b/i;
  let cur = null;
  let curCh = null;
  const hasFail = new Map();
  const hasStatus = new Map();
  const channelFails = new Map(); // label -> Set(channel)
  let hits = 0;

  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    const m = line.match(atePat);
    if (m) {
      cur = Number(m[1]);
      curCh = m[2] != null ? Number(m[2]) : null;
      continue;
    }
    const sm = line.match(ateStatus);
    if (!sm || cur == null) continue;
    hits += 1;
    hasStatus.set(cur, true);
    if (String(sm[1]).toUpperCase() === "F") {
      hasFail.set(cur, true);
      if (curCh != null) {
        if (!channelFails.has(cur)) channelFails.set(cur, new Set());
        channelFails.get(cur).add(curCh);
      } else {
        // STATUS:F without CH on same header line — still count one channel hit
        if (!channelFails.has(cur)) channelFails.set(cur, new Set());
        channelFails.get(cur).add(-1);
      }
    }
  }

  if (!hits) return null;

  const counts = {};
  for (const label of hasStatus.keys()) {
    const pid = logLabelToPatternId(label);
    if (!counts[pid]) {
      counts[pid] = {
        fails: 0,
        passes: 0,
        channel_fails: 0,
        fail_channels: [],
      };
    }
    if (hasFail.get(label)) {
      counts[pid].fails += 1;
      const chs = channelFails.get(label);
      if (chs) {
        counts[pid].channel_fails += chs.size;
        counts[pid].fail_channels = [...chs].filter((c) => c >= 0);
      }
    } else {
      counts[pid].passes += 1;
    }
  }
  return counts;
}

/**
 * Analyze log files entirely in the browser (no upload).
 * Client keeps their files local — works over the share link.
 */
async function analyzeLogsInBrowser(logItems, selectedIds, discardedIds, onProgress) {
  const selected = (selectedIds || []).map(Number);
  const discarded = (discardedIds || []).map(Number);
  const known = new Set([...selected, ...discarded]);

  const perPattern = new Map();
  const ensure = (pid) => {
    if (!perPattern.has(pid)) {
      perPattern.set(pid, {
        fails: 0,
        passes: 0,
        by_folder: {},
        files: [],
        dies_failed: new Set(),
        channel_fails: 0,
        fail_channels: new Set(),
        recent_fail_mass: 0,
      });
    }
    return perPattern.get(pid);
  };

  const foldersSeen = new Set();
  let filesParsed = 0;
  let filesWithStatus = 0;
  let filesSkipped = 0;

  const usable = (logItems || []).filter((item) => {
    const name = (item.rel || item.file?.name || "").toLowerCase();
    return (
      name.endsWith(".log") ||
      name.endsWith(".txt") ||
      name.endsWith(".csv") ||
      name.endsWith(".out") ||
      name.endsWith(".rpt")
    );
  });

  // First pass: collect folder names so drift weights are stable
  for (const item of usable) {
    const rel = (item.rel || item.file.name || "").replace(/\\/g, "/");
    foldersSeen.add(folderKeyFromRel(rel));
  }
  const folderOrder = [...foldersSeen].sort();
  const folderIdx = Object.fromEntries(folderOrder.map((f, i) => [f, i]));
  const nFold = Math.max(folderOrder.length, 1);

  for (let i = 0; i < usable.length; i++) {
    const item = usable[i];
    const rel = (item.rel || item.file.name || "").replace(/\\/g, "/");
    onProgress?.(i + 1, usable.length, rel);

    let text = "";
    try {
      text = await item.file.text();
    } catch {
      filesSkipped += 1;
      continue;
    }

    filesParsed += 1;
    const fileCounts = parseAteLogText(text);
    if (!fileCounts) continue;
    filesWithStatus += 1;
    const folder = folderKeyFromRel(rel);
    const fIdx = folderIdx[folder] ?? 0;
    const drift =
      nFold <= 1 ? 1 : 0.35 + 0.65 * Math.pow(2, (fIdx / (nFold - 1) - 1) * 1.5);

    for (const [pidStr, cp] of Object.entries(fileCounts)) {
      const pid = Number(pidStr);
      const row = ensure(pid);
      row.fails += cp.fails;
      row.passes += cp.passes;
      row.channel_fails += cp.channel_fails || 0;
      for (const ch of cp.fail_channels || []) row.fail_channels.add(ch);
      if (!row.by_folder[folder]) row.by_folder[folder] = { fails: 0, passes: 0 };
      row.by_folder[folder].fails += cp.fails;
      row.by_folder[folder].passes += cp.passes;
      if (cp.fails > 0) {
        row.dies_failed.add(folder);
        row.recent_fail_mass += drift * cp.fails;
      }
      if (cp.fails || cp.passes) {
        row.files.push({ path: rel, fails: cp.fails, passes: cp.passes });
      }
    }

    // Yield so UI stays responsive on large folders
    if (i % 2 === 1) await new Promise((r) => setTimeout(r, 0));
  }

  const serialize = (ids) =>
    ids
      .map((pid) => {
        const row = perPattern.get(pid);
        const fails = row ? row.fails : 0;
        const passes = row ? row.passes : 0;
        const total = fails + passes;
        return {
          pattern_id: pid,
          fails,
          passes,
          total,
          fail_rate: fails / Math.max(total, 1),
          by_folder: row ? row.by_folder : {},
          files: row ? row.files.slice(0, 20) : [],
          dies_failed: row ? [...row.dies_failed] : [],
          channel_fails: row ? row.channel_fails : 0,
          fail_channels: row ? [...row.fail_channels].sort((a, b) => a - b) : [],
          recent_fail_mass: row
            ? row.recent_fail_mass / Math.max(fails + passes, 1)
            : 0,
        };
      })
      .sort((a, b) => b.fails - a.fails || a.pattern_id - b.pattern_id);

  const kept = serialize(selected);
  const discardedRows = serialize(discarded);
  const unmatched = [];
  for (const [pid, row] of perPattern.entries()) {
    if (known.has(pid)) continue;
    if (!row.fails && !row.passes) continue;
    unmatched.push({ pattern_id: pid, fails: row.fails, passes: row.passes });
  }
  unmatched.sort((a, b) => b.fails - a.fails);

  const keptFails = kept.reduce((s, r) => s + r.fails, 0);
  const discFails = discardedRows.reduce((s, r) => s + r.fails, 0);

  return {
    folders: [...foldersSeen].sort(),
    n_folders: foldersSeen.size,
    files_parsed: filesParsed,
    files_with_status: filesWithStatus,
    files_skipped: filesSkipped,
    id_mapping: "log_Pn -> pattern_id n-1 (browser-local analysis)",
    summary: {
      kept_n: selected.length,
      discarded_n: discarded.length,
      kept_with_fails: kept.filter((r) => r.fails > 0).length,
      discarded_with_fails: discardedRows.filter((r) => r.fails > 0).length,
      kept_total_fails: keptFails,
      discarded_total_fails: discFails,
      all_total_fails: keptFails + discFails,
      unmatched_patterns: unmatched.length,
      unmatched_fails: unmatched.reduce((s, u) => s + u.fails, 0),
    },
    kept,
    discarded: discardedRows,
    kept_failing: kept.filter((r) => r.fails > 0),
    discarded_failing: discardedRows.filter((r) => r.fails > 0),
    unmatched: unmatched.slice(0, 100),
  };
}

function PatternListPanel({
  title,
  tone,
  ids,
  emptyText,
  example,
  detailsById,
  onSelectId,
  selectedId,
  failById,
  sortByFails,
}) {
  const list = useMemo(() => {
    const base = ids || [];
    if (!sortByFails || !failById) return base;
    return [...base].sort((a, b) => {
      const fa = failById[a]?.fails || 0;
      const fb = failById[b]?.fails || 0;
      if (fb !== fa) return fb - fa;
      return a - b;
    });
  }, [ids, failById, sortByFails]);

  return (
    <div className={`pattern-panel ${tone}`}>
      <div className="pattern-panel-head">
        <h2>{title}</h2>
        <span className="chooser-meta">{list.length} patterns</span>
      </div>
      {tone === "discard" && example && (
        <div className="discard-example">
          <div className="discard-example-title">
            Example — why P{example.pattern_id} was not loaded
          </div>
          <p className="discard-example-body">{example.reason}</p>
          <div className="discard-example-meta">
            Closest kept: <strong>P{example.nearest_kept}</strong>
            {" · "}
            embedding distance{" "}
            <strong>{example.distance_to_nearest}</strong>
            {" · "}
            diversity rank #{example.diversity_rank}
            {example.n_diff != null && (
              <>
                {" · "}
                bit distance <strong>{example.n_diff}</strong> different /{" "}
                <strong>{example.n_same}</strong> same
              </>
            )}
          </div>
          {example.bits && example.nearest_bits && (
            <div className="discard-bits-compare">
              <div className="keep-vs-skip">
                <div className="keep-vs-skip-side keep">
                  <span className="keep-vs-skip-role">Kept</span>
                  <span className="keep-vs-skip-pid">P{example.nearest_kept}</span>
                </div>
                <div className="keep-vs-skip-mid">
                  <div className="keep-vs-skip-diff">
                    <strong>{example.n_diff ?? "—"}</strong> bits different
                  </div>
                  <div className="keep-vs-skip-same">
                    <strong>{example.n_same ?? "—"}</strong> bits the same
                  </div>
                  <div className="keep-vs-skip-scores">
                    bit distance <strong>{example.n_diff ?? "—"}</strong>
                    {" · "}
                    embedding distance{" "}
                    <strong>{example.distance_to_nearest}</strong>
                  </div>
                </div>
                <div className="keep-vs-skip-side skip">
                  <span className="keep-vs-skip-role">Not loaded</span>
                  <span className="keep-vs-skip-pid">P{example.pattern_id}</span>
                </div>
              </div>
              <div className="discard-bits-caption">
                P{example.nearest_kept} was <strong>kept</strong> and P
                {example.pattern_id} was <strong>not loaded</strong> because they
                differ in only <strong>{example.n_diff ?? "—"}</strong> of 234
                bits (<strong>{example.n_same ?? "—"}</strong> the same) —
                embedding distance{" "}
                <strong>{example.distance_to_nearest}</strong>. Yellow cells mark
                those different bits.
              </div>
              <BitStrip
                bits={example.nearest_bits}
                flip={example.flip}
                label={`Kept P${example.nearest_kept}`}
              />
              <BitStrip
                bits={example.bits}
                flip={example.flip}
                label={`Not loaded P${example.pattern_id}`}
              />
            </div>
          )}
          <div className="discard-example-hint">
            Click any low-risk chip below to compare its bits with the closest
            kept pattern.
          </div>
        </div>
      )}
      <div className="pattern-panel-body">
        {list.length === 0 ? (
          <div className="how-pick-idle">{emptyText || "—"}</div>
        ) : (
          <div className="pattern-id-grid">
            {list.map((pid, i) => {
              const active = selectedId === pid;
              const fails = failById?.[pid]?.fails;
              const detail = detailsById?.[pid];
              const embDist = detail?.distance_to_nearest;
              const Chip = onSelectId ? "button" : "span";
              return (
                <Chip
                  type={onSelectId ? "button" : undefined}
                  key={`${tone}-${pid}-${i}`}
                  className={`pattern-id-chip ${active ? "active" : ""} ${
                    fails > 0 ? "has-fail" : ""
                  }`}
                  onClick={onSelectId ? () => onSelectId(pid) : undefined}
                  title={
                    fails != null
                      ? `P${pid}: ${fails} fail(s)` +
                        (detail?.reason ? ` — ${detail.reason}` : "")
                      : detail?.reason ||
                        (tone === "discard"
                          ? "Click for why discarded"
                          : undefined)
                  }
                >
                  <span className="pattern-rank">#{i + 1}</span>
                  <span className="pattern-id-main">P{pid}</span>
                  {embDist != null && (
                    <span className="emb-dist" title="Embedding distance to closest kept">
                      d={embDist}
                    </span>
                  )}
                  {fails > 0 && <span className="fail-badge">{fails}F</span>}
                </Chip>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}



function tierLabel(tier) {
  if (tier === "log_fail_greedy") return "log fail";
  if (tier === "log_fail") return "log fail";
  if (tier === "ml_unseen") return "ml unseen";
  return tier || "—";
}

function NextChipRecommend({ nextChip }) {
  if (!nextChip?.recommend?.length) return null;

  const copyList = () => {
    const text = nextChip.recommend.map((r) => `P${r.pattern_id}`).join(", ");
    try {
      navigator.clipboard?.writeText(text);
    } catch {
      /* ignore */
    }
  };

  const nLogFail = nextChip.n_log_fail ?? 0;
  const nMl = nextChip.n_ml_unseen ?? 0;

  return (
    <div className="next-chip">
      <div className="next-chip-head">
        <div>
          <h2>Recommended patterns for next chip</h2>
          <p className="field-hint">
            From your logs only: <strong>1 log = 1 die/chip</strong>. Every
            pattern that failed on that chip is included, ordered by severity
            (channels → fail rate → fails). Patterns that only passed are
            skipped.
          </p>
        </div>
        <button type="button" className="btn secondary" onClick={copyList}>
          Copy ordered list
        </button>
      </div>

      <div className="fail-summary next-chip-metrics">
        <div className="metric">
          <div className="k">Recommend</div>
          <div className="v">
            {nextChip.recommend.length}
            <span className="v-sub"> / {nextChip.n_patterns_total}</span>
          </div>
        </div>
        <div className="metric">
          <div className="k">Failed in logs</div>
          <div className="v">
            {nextChip.n_failing_patterns ?? nLogFail}
            <span className="v-sub"> must-run</span>
          </div>
        </div>
        <div className="metric">
          <div className="k">Log pass → skip</div>
          <div className="v">{nextChip.n_log_pass_skip ?? "—"}</div>
        </div>
        <div className="metric">
          <div className="k">Log fail / ML unseen</div>
          <div className="v" style={{ fontSize: "0.95rem" }}>
            {nLogFail} · {nMl}
          </div>
        </div>
      </div>

      <div className="next-chip-scroll">
        <table className="fail-table next-chip-table">
          <thead>
            <tr>
              <th>Order</th>
              <th>Pattern</th>
              <th>Tier</th>
              <th>Fail rate</th>
              <th>Fails</th>
              <th>Why</th>
            </tr>
          </thead>
          <tbody>
            {nextChip.recommend.map((r) => (
              <tr
                key={`next-${r.pattern_id}-${r.rank}`}
                className={`tier-${r.tier}`}
              >
                <td>{r.rank}</td>
                <td>
                  P{r.pattern_id}
                  {r.discarded ? (
                    <span className="tier-pill skip-pill">was skip</span>
                  ) : null}
                </td>
                <td>
                  <span className={`tier-pill tier-${r.tier}`}>
                    {tierLabel(r.tier)}
                  </span>
                </td>
                <td>{(r.fail_rate * 100).toFixed(1)}%</td>
                <td className={r.fails ? "fail-num" : ""}>{r.fails}</td>
                <td className="why-cell">{(r.reasons || []).join(" · ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatTestTime(ms) {
  if (ms == null || Number.isNaN(Number(ms))) return "—";
  const n = Number(ms);
  if (n >= 1000) return `${(n / 1000).toFixed(2)} s`;
  if (n >= 1) return `${n.toFixed(2)} ms`;
  return `${n.toFixed(4)} ms`;
}

function RamBar({ value, max, tone }) {
  const pct = Math.min(100, (value / Math.max(max, 1e-6)) * 100);
  return (
    <div className="bar-track">
      <div className={`bar-fill ${tone}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function TenSlots({ title, tone, slots, cursor, selectedStep, onSelect }) {
  return (
    <div className={`ten-board ${tone}`}>
      <div className="ten-title">{title}</div>
      <div className="ten-grid">
        {Array.from({ length: FIRST_N }, (_, i) => {
          const slot = slots[i];
          const filled = Boolean(slot);
          const active = cursor === i + 1;
          const viewing = selectedStep === i + 1;
          return (
            <button
              type="button"
              key={`${tone}-slot-${i}`}
              className={[
                "ten-slot",
                filled ? "filled" : "empty",
                active ? "active" : "",
                viewing ? "viewing" : "",
                slot?.action || "",
              ].join(" ")}
              disabled={!filled || !onSelect}
              onClick={() => filled && onSelect?.(i + 1)}
              title={
                filled
                  ? `View how slot ${i + 1} picked P${slot.pattern_id}`
                  : `Slot ${i + 1} empty`
              }
            >
              <div className="ten-slot-num">{i + 1}</div>
              <div className="ten-slot-pid">
                {filled ? `P${slot.pattern_id}` : "—"}
              </div>
              <div className="ten-slot-tag">
                {filled ? String(slot.action || "").toUpperCase() : ""}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function BitStrip({ bits, flip, label, markSame, diffClass = "diff" }) {
  const list = bits || [];
  return (
    <div className="bit-strip-wrap">
      {label && <span className="bit-strip-label">{label}</span>}
      <div className="bit-strip">
        {list.map((b, i) => {
          const isDiff = flip && flip[i];
          const isSame = markSame && flip && !flip[i];
          return (
            <span
              key={`b-${i}`}
              className={[
                "bit-cell",
                b === 1 ? "one" : "zero",
                isDiff ? diffClass : "",
                isSame ? "same" : "",
              ].join(" ")}
            >
              {b}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function HowPickVisual({
  step,
  pick,
  board,
  already,
  refBits,
  refPid,
  maxStep,
  onPrev,
  onNext,
}) {
  if (!step || step > FIRST_N) {
    return (
      <div className="how-pick idle">
        <div className="how-pick-label">HOW VERILUMEN AGENT TAKES (0 / 1 BITS)</div>
        <div className="how-pick-idle">
          Full scan chain = 234 bits. Yellow = different from already-kept. Most
          different → TAKE. Click any filled Verilumen agent slot to replay that step.
        </div>
      </div>
    );
  }

  const winner = (board || []).find((b) => b.winner);
  const isSeed = Boolean(winner?.seed) || step === 1;
  const prevPid = already?.length ? already[already.length - 1] : null;

  return (
    <div className="how-pick">
      <div className="how-pick-nav">
        <button
          type="button"
          className="btn secondary how-nav-btn"
          disabled={step <= 1}
          onClick={onPrev}
        >
          ← Prev
        </button>
        <div className="how-pick-label">
          HOW VERILUMEN AGENT TAKES SLOT {step}
          {winner ? ` → P${winner.pattern_id}` : ""}
          {isSeed
            ? " — start seed"
            : prevPid != null
              ? ` — after P${prevPid} (highest distance wins)`
              : " — highest Verilumen agent distance wins"}
          <span className="how-step-hint">
            {" "}
            Distance = how far this pattern’s Verilumen agent embedding is from
            already kept. Click slots to jump.
          </span>
        </div>
        <button
          type="button"
          className="btn secondary how-nav-btn"
          disabled={!maxStep || step >= maxStep}
          onClick={onNext}
        >
          Next →
        </button>
      </div>

      <div className="how-bits-layout">
        <div className="how-bits-side">
          <div className="how-col-title">Already taken (before this slot)</div>
          <div className="how-chips">
            {(already || []).length === 0 && (
              <span className="how-chip empty">none yet</span>
            )}
            {(already || []).map((pid) => (
              <span
                key={`al-${pid}`}
                className={`how-chip ${pid === refPid ? "ref" : ""}`}
              >
                P{pid}
              </span>
            ))}
          </div>
          {refBits && (
            <BitStrip
              bits={refBits}
              label={refPid != null ? `Compare vs P${refPid}` : "Compare vs"}
            />
          )}
        </div>

        <div className="how-bits-main">
          <div className="how-col-title">
            {isSeed
              ? "Seed: full 234 scan-in bits (channel 1)"
              : "Winner only — highest Verilumen agent distance (yellow bits = view)"}
          </div>
          <div className="how-bit-rows">
            {(board || [])
              .filter((b) => b.winner)
              .map((b) => (
              <div
                key={`hb-${b.pattern_id}-${b.winner ? "w" : "r"}`}
                className={`how-bit-row ${b.winner ? "win" : ""}`}
              >
                <span className="how-bar-pid">P{b.pattern_id}</span>
                <BitStrip bits={b.bits} flip={b.flip} />
                <div className="how-bit-score">
                  <div className="how-dist">
                    {b.seed
                      ? "seed (no score)"
                      : `Verilumen agent distance ${Number(b.min_dist ?? 0).toFixed(4)}`}
                  </div>
                  <div className="how-bits-meta">
                    {b.n_diff != null
                      ? `${b.n_diff} different / ${b.n_same ?? 0} same (view)`
                      : "234-bit view"}
                  </div>
                </div>
                {b.winner && (
                  <span className="how-win-tag">{isSeed ? "SEED" : "TAKE"}</span>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="how-col result">
          <div className="how-col-title">Goes to slot {step}</div>
          <div className={`how-result-card ${pick?.action || "keep"}`}>
            <div className="how-result-pid">
              {pick ? `P${pick.pattern_id}` : "—"}
            </div>
            <div className="how-result-tag">
              {pick ? String(pick.action).toUpperCase() : ""}
            </div>
            {winner?.bits && <BitStrip bits={winner.bits} flip={winner.flip} />}
          </div>
        </div>
      </div>
    </div>
  );
}

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const line = chunk
        .split("\n")
        .map((l) => l.trim())
        .find((l) => l.startsWith("data:"));
      if (!line) continue;
      const payload = line.replace(/^data:\s*/, "");
      try {
        onEvent(JSON.parse(payload));
      } catch {
        /* ignore malformed */
      }
    }
  }
}

export default function App() {
  const [theme, setTheme] = useState(readStoredTheme);
  const [processMode, setProcessMode] = useState("pre"); // pre = STIL predict, post = log analysis
  const [file, setFile] = useState(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Upload a STIL file, then click Run live simulation.");
  const [statusTone, setStatusTone] = useState("");
  const [profile, setProfile] = useState(null);
  const [series, setSeries] = useState([]);
  const [fullMb, setFullMb] = useState(0);
  const [lstmMb, setLstmMb] = useState(0);
  const [rssMb, setRssMb] = useState(0);
  const [progress, setProgress] = useState(0);
  const [done, setDone] = useState(null);
  const [discardFocusId, setDiscardFocusId] = useState(null);
  const [normalSlots, setNormalSlots] = useState([]);
  const [lstmSlots, setLstmSlots] = useState([]);
  const [focusStep, setFocusStep] = useState(0);
  const [viewStep, setViewStep] = useState(0);
  const [howHistory, setHowHistory] = useState([]); // index 0 = slot 1
  const [followLive, setFollowLive] = useState(true);
  const followLiveRef = useRef(true);
  const abortRef = useRef(null);
  const [logFiles, setLogFiles] = useState([]); // { file, rel }
  const [logFolders, setLogFolders] = useState([]);
  const [failAnalysis, setFailAnalysis] = useState(null);
  const [analyzingFails, setAnalyzingFails] = useState(false);
  const logInputRef = useRef(null);
  const logFilesInputRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const chartColors = useMemo(
    () =>
      theme === "light"
        ? {
            grid: "#d0d7de",
            tick: "#5a6570",
            tooltipBg: "#ffffff",
            tooltipBorder: "#d0d7de",
            tooltipText: "#1a1f26",
            full: "#3d6ea8",
            lstm: "#1f8f5f",
            rss: "#2b7fd4",
          }
        : {
            grid: "#2a3340",
            tick: "#8b98a8",
            tooltipBg: "#161b22",
            tooltipBorder: "#2a3340",
            tooltipText: "#e8edf4",
            full: "#5b8fd4",
            lstm: "#3ecf8e",
            rss: "#4ea1f3",
          },
    [theme]
  );

  const gaugeMax = useMemo(() => {
    const peak = profile?.full_peak_mb || Math.max(fullMb, lstmMb, 1);
    return Math.max(peak, fullMb, lstmMb, 1);
  }, [profile, fullMb, lstmMb]);

  const timeGaugeMax = useMemo(() => {
    if (!done) return 1;
    return Math.max(Number(done.full_ms) || 0, Number(done.lstm_ms) || 0, 1);
  }, [done]);

  const savedTestMs = useMemo(() => {
    if (!done) return null;
    const full = Number(done.full_ms) || 0;
    const sub = Number(done.lstm_ms) || 0;
    return {
      ms: full - sub,
      pct: full > 0 ? (100 * (full - sub)) / full : 0,
    };
  }, [done]);

  const howView = useMemo(() => {
    if (viewStep < 1) return null;
    return howHistory[viewStep - 1] || null;
  }, [howHistory, viewStep]);

  const maxFilledStep = useMemo(() => {
    let m = 0;
    for (let i = 0; i < howHistory.length; i++) {
      if (howHistory[i]) m = i + 1;
    }
    return m;
  }, [howHistory]);

  const resetLive = () => {
    setSeries([]);
    setFullMb(0);
    setLstmMb(0);
    setRssMb(0);
    setProgress(0);
    setDone(null);
    setDiscardFocusId(null);
    setProfile(null);
    setStatusTone("");
    setNormalSlots([]);
    setLstmSlots([]);
    setFocusStep(0);
    setViewStep(0);
    setHowHistory([]);
    followLiveRef.current = true;
    setFollowLive(true);
    setFailAnalysis(null);
  };

  const addLogFolder = (fileList) => {
    const incoming = Array.from(fileList || []);
    if (!incoming.length) return;
    const next = [...logFiles];
    const folderSet = new Set(logFolders);
    for (const f of incoming) {
      const rel = (f.webkitRelativePath || f.name || "").replace(/\\/g, "/");
      if (!rel) continue;
      const top = rel.split("/")[0] || rel;
      folderSet.add(top);
      const key = rel.toLowerCase();
      if (next.some((x) => x.rel.toLowerCase() === key)) continue;
      next.push({ file: f, rel });
    }
    setLogFiles(next);
    setLogFolders([...folderSet].sort());
    setFailAnalysis(null);
  };

  const clearLogs = () => {
    setLogFiles([]);
    setLogFolders([]);
    setFailAnalysis(null);
    if (logInputRef.current) logInputRef.current.value = "";
    if (logFilesInputRef.current) logFilesInputRef.current.value = "";
  };

  const analyzeFails = async () => {
    if (analyzingFails) return;
    if (!done?.selected_ids?.length && !done?.discarded_ids?.length) {
      setStatus("Run a live simulation first so kept/discarded patterns exist.");
      setStatusTone("error");
      return;
    }
    if (!logFiles.length) {
      setStatus("Add one or more log folders first.");
      setStatusTone("error");
      return;
    }

    setAnalyzingFails(true);
    setStatusTone("");
    setStatus(
      `Reading local logs on this PC (${logFolders.length} folder(s) / ${logFiles.length} file(s))…`
    );

    try {
      // Analyze on the client's machine — no large upload through the share link
      const data = await analyzeLogsInBrowser(
        logFiles,
        done.selected_ids || [],
        done.discarded_ids || [],
        (n, total, rel) => {
          setStatus(`Parsing log ${n}/${total}: ${rel}`);
        }
      );

      let nextChip = null;
      if (data.files_with_status) {
        setStatus("Running greedy + ML next-chip ranking…");
        nextChip = recommendNextChipGreedyMl(data, done);
        data.next_chip = nextChip;
      }

      setFailAnalysis(data);
      const s = data.summary || {};
      if (!data.files_with_status) {
        setStatusTone("error");
        setStatus(
          "No STATUS:F / STATUS:P lines found in the selected logs. Use Advantest pattern execution logs."
        );
      } else {
        setStatusTone("ok");
        const nRec = nextChip?.recommend?.length ?? 0;
        const nFail = nextChip?.n_failing_patterns ?? 0;
        const nSkipPass = nextChip?.n_log_pass_skip ?? 0;
        setStatus(
          `Next-chip: ${nRec} to run (${nFail} failed in logs, ` +
            `${nextChip?.n_ml_unseen ?? 0} ML not-in-log) · ` +
            `${nSkipPass} log pass-only skipped`
        );
      }
    } catch (err) {
      setStatus(err.message || String(err));
      setStatusTone("error");
    } finally {
      setAnalyzingFails(false);
    }
  };

  const jumpToStep = (step) => {
    if (step < 1 || step > FIRST_N) return;
    if (!howHistory[step - 1]) return;
    followLiveRef.current = false;
    setFollowLive(false);
    setViewStep(step);
  };

  const onEvent = (ev) => {
    if (ev.type === "status" || ev.type === "log" || ev.type === "job") {
      if (ev.message) setStatus(ev.message);
      return;
    }
    if (ev.type === "error") {
      setStatus(ev.message || "Simulation failed");
      setStatusTone("error");
      setRunning(false);
      return;
    }
    if (ev.type === "await_seed") {
      // legacy: ignore interactive seed prompts
      return;
    }
    if (ev.type === "profile") {
      setProfile(ev);
      setStatus(
        `Profile: ${ev.n_patterns} patterns · ${ev.n_pins} pins · ${ev.total_cycles?.toLocaleString?.() || ev.total_cycles} cycles`
      );
      return;
    }
    if (ev.type === "chooser") {
      const step = ev.step || 0;
      setFocusStep(Math.min(step, FIRST_N));
      if (ev.message) setStatus(ev.message);
      // Only visualize first 10 picks
      if (step >= 1 && step <= FIRST_N) {
        setProgress(step / FIRST_N);
        const n = ev.normal;
        const l = ev.lstm;
        if (n?.pattern_id != null) {
          setNormalSlots((prev) => {
            const next = prev.slice();
            next[step - 1] = {
              pattern_id: n.pattern_id,
              action: "run",
            };
            return next;
          });
        }
        if (l?.pattern_id != null) {
          const pick = {
            pattern_id: l.pattern_id,
            action: l.action || "keep",
            gain: l.gain,
          };
          setLstmSlots((prev) => {
            const next = prev.slice();
            next[step - 1] = pick;
            return next;
          });
          const soFar = l.selected_so_far || [];
          const snap = {
            pick,
            board: l.compare_board || [],
            already: soFar.filter((pid) => pid !== l.pattern_id).slice(-8),
            refBits: l.ref_bits || null,
            refPid: l.ref_pattern_id != null ? l.ref_pattern_id : null,
          };
          setHowHistory((prev) => {
            const next = prev.slice();
            next[step - 1] = snap;
            return next;
          });
          if (followLiveRef.current) setViewStep(step);
        }
      }
      return;
    }
    if (ev.type === "progress") {
      setFullMb(ev.full_mb ?? 0);
      setLstmMb(ev.lstm_mb ?? 0);
      setRssMb(ev.rss_mb ?? 0);
      if (ev.phase !== "select") {
        setProgress(ev.total ? ev.step / ev.total : 0);
      }
      if (ev.message) setStatus(ev.message);
      setSeries((prev) => {
        const next = [
          ...prev,
          {
            step: ev.step,
            phase: ev.phase,
            full_mb: ev.full_mb ?? 0,
            lstm_mb: ev.lstm_mb ?? 0,
            rss_mb: ev.rss_mb ?? 0,
          },
        ];
        return next.length > 800 ? next.slice(-800) : next;
      });
      return;
    }
    if (ev.type === "done") {
      setDone(ev);
      setDiscardFocusId(ev.discard_example?.pattern_id ?? ev.discarded_ids?.[0] ?? null);
      setFullMb(ev.full_peak_mb);
      setLstmMb(ev.lstm_peak_mb);
      setProgress(1);
      setStatusTone("ok");
      setStatus(
        `Done. Verilumen agent chose ${ev.selected_n}/${ev.total_n} patterns — ${ev.saved_pct}% less peak vector RAM.`
      );
      setRunning(false);
    }
  };

  const run = async () => {
    if (running) return;
    if (!file) {
      setStatus("Upload a STIL file first.");
      setStatusTone("error");
      return;
    }
    resetLive();
    setRunning(true);
    setStatus("Starting simulation…");

    const body = new FormData();
    body.append("stil", file);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API}/api/simulate`, {
        method: "POST",
        body,
        signal: controller.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      await readSse(res, onEvent);
      setRunning(false);
    } catch (err) {
      if (err.name === "AbortError") {
        setStatus("Simulation stopped.");
      } else {
        setStatus(err.message || String(err));
        setStatusTone("error");
      }
      setRunning(false);
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setRunning(false);
  };

  const loadSeries = series.filter((d) => d.phase === "load" || d.phase === "embed");

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-copy">
          <h1>Test Time and Vector Memory Optimization — Verilumen Simulation Agent</h1>
          <p>
            Pre-process: STIL → pattern prediction. Post-process: ATE logs →
            fail analysis. Click any filled Verilumen agent slot (or Prev/Next)
            to replay how that pick was chosen — yellow cells = different 0/1
            bits.
          </p>
          <div className="flow">
            Pre: STIL → predict · Post: logs → fail analysis
          </div>
        </div>
        <div className="theme-toggle" role="group" aria-label="Color theme">
          <button
            type="button"
            className={theme === "light" ? "active" : ""}
            onClick={() => setTheme("light")}
            aria-pressed={theme === "light"}
          >
            Light
          </button>
          <button
            type="button"
            className={theme === "dark" ? "active" : ""}
            onClick={() => setTheme("dark")}
            aria-pressed={theme === "dark"}
          >
            Dark
          </button>
        </div>
      </header>

      <div className="process-switch" role="tablist" aria-label="Process stage">
        <button
          type="button"
          role="tab"
          className={processMode === "pre" ? "active" : ""}
          aria-selected={processMode === "pre"}
          onClick={() => setProcessMode("pre")}
        >
          <span className="process-k">Pre-process</span>
          <span className="process-v">STIL → pattern prediction</span>
        </button>
        <button
          type="button"
          role="tab"
          className={processMode === "post" ? "active" : ""}
          aria-selected={processMode === "post"}
          onClick={() => setProcessMode("post")}
        >
          <span className="process-k">Post-process</span>
          <span className="process-v">ATE logs → fail analysis</span>
        </button>
      </div>

      {processMode === "pre" && (
      <div className="layout">
        <aside className="panel">
          <div className="process-panel-tag">Pre-process</div>
          <h2>STIL file</h2>
          <p className="field-hint">
            Before ATE: upload a STIL and run the Verilumen agent to predict
            which patterns to load and which to skip.
          </p>
          <div className="field">
            <label htmlFor="stil">Upload</label>
            <input
              id="stil"
              type="file"
              accept=".stil,.STIL,.txt"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            {file && <div className="file-name">{file.name}</div>}
          </div>

          <button className="btn" onClick={run} disabled={running || !file}>
            {running ? "Running…" : "Run live simulation"}
          </button>
          <button className="btn secondary" onClick={stop} disabled={!running}>
            Stop
          </button>
        </aside>

        <main>
          <div className={`status ${statusTone}`}>{status}</div>
          <div className="progress">
            <div style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>

          <div className="chooser-window">
            <div className="chooser-window-head">
              <h2>First 10 patterns</h2>
              <div className="chooser-meta">
                {maxFilledStep > 0
                  ? `Viewing slot ${viewStep || "—"} / ${maxFilledStep}${
                      !followLive ? " (paused — click Next or Live)" : ""
                    }`
                  : "—"}
                {maxFilledStep > 0 && !followLive && (
                  <button
                    type="button"
                    className="btn secondary how-nav-btn"
                    style={{ marginLeft: 8 }}
                    onClick={() => {
                      followLiveRef.current = true;
                      setFollowLive(true);
                      setViewStep(maxFilledStep);
                    }}
                  >
                    Live
                  </button>
                )}
              </div>
            </div>
            <div className="ten-wrap">
              <TenSlots
                title="without verilumen agent order"
                tone="normal"
                slots={normalSlots}
                cursor={focusStep}
                selectedStep={viewStep}
                onSelect={jumpToStep}
              />
              <TenSlots
                title="top 10 pick of verilumen agent"
                tone="lstm"
                slots={lstmSlots}
                cursor={focusStep}
                selectedStep={viewStep}
                onSelect={jumpToStep}
              />
            </div>
            <HowPickVisual
              step={howView ? viewStep : 0}
              pick={howView?.pick}
              board={howView?.board}
              already={howView?.already}
              refBits={howView?.refBits}
              refPid={howView?.refPid}
              maxStep={maxFilledStep}
              onPrev={() => jumpToStep(viewStep - 1)}
              onNext={() => jumpToStep(viewStep + 1)}
            />
          </div>

          <div className="gauges">
            <div className="gauge-card full">
              <div className="label">without verilumen agent vector RAM</div>
              <div className="value">{fullMb.toFixed(2)} MB</div>
              <RamBar value={fullMb} max={gaugeMax} tone="full" />
            </div>
            <div className="gauge-card lstm">
              <div className="label">with verilumen agent vector RAM</div>
              <div className="value">{lstmMb.toFixed(2)} MB</div>
              <RamBar value={lstmMb} max={gaugeMax} tone="lstm" />
            </div>
          </div>

          <div className="gauges">
            <div className="gauge-card full">
              <div className="label">without verilumen agent test time</div>
              <div className="value">
                {done ? formatTestTime(done.full_ms) : "—"}
              </div>
              <RamBar
                value={done ? Number(done.full_ms) || 0 : 0}
                max={timeGaugeMax}
                tone="full"
              />
            </div>
            <div className="gauge-card lstm">
              <div className="label">with verilumen agent test time</div>
              <div className="value">
                {done ? formatTestTime(done.lstm_ms) : "—"}
              </div>
              <RamBar
                value={done ? Number(done.lstm_ms) || 0 : 0}
                max={timeGaugeMax}
                tone="lstm"
              />
            </div>
          </div>

          <div className="metrics">
            <div className="metric">
              <div className="k">Host RSS</div>
              <div className="v">{rssMb ? `${rssMb.toFixed(0)} MB` : "—"}</div>
            </div>
            <div className="metric">
              <div className="k">Patterns</div>
              <div className="v">{profile?.n_patterns ?? "—"}</div>
            </div>
            <div className="metric">
              <div className="k">Pins</div>
              <div className="v">{profile?.n_pins ?? "—"}</div>
            </div>
            <div className="metric">
              <div className="k">Cycles</div>
              <div className="v">
                {profile?.total_cycles?.toLocaleString?.() ?? "—"}
              </div>
            </div>
            <div className="metric">
              <div className="k">Test time saved</div>
              <div className="v">
                {savedTestMs
                  ? `${formatTestTime(savedTestMs.ms)} (${savedTestMs.pct.toFixed(1)}%)`
                  : "—"}
              </div>
            </div>
          </div>

          <div className="chart-wrap">
            <div className="chart-title">
              Live vector memory (MB) — blue = without verilumen agent, green =
              with verilumen agent
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={loadSeries} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <defs>
                  <linearGradient id="gFull" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={chartColors.full} stopOpacity={0.45} />
                    <stop offset="100%" stopColor={chartColors.full} stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gLstm" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={chartColors.lstm} stopOpacity={0.4} />
                    <stop offset="100%" stopColor={chartColors.lstm} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  stroke={chartColors.tick}
                  tick={{ fill: chartColors.tick, fontSize: 11 }}
                />
                <YAxis
                  yAxisId="ram"
                  stroke={chartColors.tick}
                  tick={{ fill: chartColors.tick, fontSize: 11 }}
                />
                <YAxis
                  yAxisId="rss"
                  orientation="right"
                  stroke={chartColors.rss}
                  tick={{ fill: chartColors.rss, fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    background: chartColors.tooltipBg,
                    border: `1px solid ${chartColors.tooltipBorder}`,
                    color: chartColors.tooltipText,
                  }}
                />
                <Legend />
                <Area
                  yAxisId="ram"
                  type="monotone"
                  dataKey="full_mb"
                  name="without verilumen agent"
                  stroke={chartColors.full}
                  fill="url(#gFull)"
                  strokeWidth={2.5}
                  isAnimationActive={false}
                />
                <Area
                  yAxisId="ram"
                  type="monotone"
                  dataKey="lstm_mb"
                  name="with verilumen agent"
                  stroke={chartColors.lstm}
                  fill="url(#gLstm)"
                  strokeWidth={2.5}
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="rss"
                  type="monotone"
                  dataKey="rss_mb"
                  name="Host RSS"
                  stroke={chartColors.rss}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {done && (
            <div className="done-banner">
              Verilumen agent chose{" "}
              <strong>
                {done.selected_n}/{done.total_n}
              </strong>{" "}
              patterns
              {done.reason ? <> — {done.reason}</> : null}
              <br />
              Without verilumen agent <strong>{done.full_peak_mb} MB</strong> /
              test time <strong>{formatTestTime(done.full_ms)}</strong>
              {" → "}
              With verilumen agent <strong>{done.lstm_peak_mb} MB</strong> /
              test time <strong>{formatTestTime(done.lstm_ms)}</strong>
              {" · "}
              saved <strong>{done.saved_mb} MB</strong> ({done.saved_pct}%)
              {savedTestMs ? (
                <>
                  {" · "}
                  test time saved{" "}
                  <strong>
                    {formatTestTime(savedTestMs.ms)} ({savedTestMs.pct.toFixed(1)}
                    %)
                  </strong>
                </>
              ) : null}
            </div>
          )}

          {done && (
            <div className="lists-wrap">
              <PatternListPanel
                title="high recommended patterns order"
                tone="keep"
                ids={done.selected_ids}
                emptyText="No kept patterns"
                detailsById={Object.fromEntries(
                  (done.selected_details || []).map((d) => [d.pattern_id, d])
                )}
              />
              <PatternListPanel
                title="low risk patterns"
                tone="discard"
                ids={done.discarded_ids}
                emptyText="Nothing discarded"
                example={
                  (done.discarded_details || []).find(
                    (d) => d.pattern_id === discardFocusId
                  ) || done.discard_example
                }
                detailsById={Object.fromEntries(
                  (done.discarded_details || []).map((d) => [d.pattern_id, d])
                )}
                selectedId={discardFocusId}
                onSelectId={setDiscardFocusId}
              />
            </div>
          )}
        </main>
      </div>
      )}

      {processMode === "post" && (
      <div className="layout">
        <aside className="panel">
          <div className="process-panel-tag post">Post-process</div>
          <h2>Fail logs</h2>
          <p className="field-hint">
            After ATE: one log file = one die/chip. Analysis stays on this PC.
            Add the log from chip N to recommend patterns for chip N+1.
          </p>
          <input
            id="logs-folder"
            ref={logInputRef}
            type="file"
            multiple
            className="sr-only-file"
            onChange={(e) => {
              addLogFolder(e.target.files);
              e.target.value = "";
            }}
            {...{ webkitdirectory: "true", directory: "true" }}
          />
          <input
            id="logs-files"
            ref={logFilesInputRef}
            type="file"
            multiple
            accept=".log,.txt,.csv,.out,.rpt,.sum,.dat,text/*"
            className="sr-only-file"
            onChange={(e) => {
              addLogFolder(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            className="btn secondary"
            onClick={() => logInputRef.current?.click()}
          >
            Add log folder
          </button>
          <button
            type="button"
            className="btn secondary"
            onClick={() => logFilesInputRef.current?.click()}
          >
            Add log files
          </button>
          {logFolders.length > 0 ? (
            <div className="log-folder-list">
              {logFolders.map((name) => (
                <div key={name} className="file-name">
                  {name}
                </div>
              ))}
              <div className="chooser-meta">
                {logFolders.length} folder(s) · {logFiles.length} file(s)
              </div>
            </div>
          ) : (
            <div className="field-hint">No logs added yet</div>
          )}
          {!done && (
            <div className="field-hint warn">
              Finish Pre-process (Run live simulation) first, then Analyze
            </div>
          )}
          <button
            type="button"
            className="btn"
            onClick={analyzeFails}
            disabled={!done || !logFiles.length || analyzingFails}
          >
            {analyzingFails ? "Analyzing…" : "Analyze & recommend next chip"}
          </button>
          <button
            type="button"
            className="btn secondary"
            onClick={clearLogs}
            disabled={!logFiles.length && !failAnalysis}
          >
            Clear logs
          </button>
        </aside>

        <main>
          <div className={`status ${statusTone || ""}`}>
            {analyzingFails
              ? status
              : failAnalysis
                ? status
                : done
                  ? `Post-process ready — ${done.total_n ?? "—"} patterns from pre-process (keep ${done.selected_n} · skip ${done.discarded_ids?.length ?? 0}). Add logs and Analyze.`
                  : "Post-process — run Pre-process first so keep / skip pattern lists exist."}
          </div>

          {!failAnalysis && (
            <div className="post-empty">
              <h2>Next-chip pattern recommendation</h2>
              <p>
                Pre-process predicts from STIL before ATE. Post-process uses
                real fail logs from tested chips, then recommends a reduced
                ordered pattern list for the next chip.
              </p>
              <ol>
                <li>Complete Pre-process (STIL → Run live simulation)</li>
                <li>Add log folders (chips/dies) from your PC</li>
                <li>Click Analyze &amp; recommend next chip</li>
              </ol>
              <p className="field-hint">
                Log fail → run on next chip. Log pass-only → skip.
              </p>
            </div>
          )}

          {failAnalysis && (
            <div className="fail-analysis">
              <div className="fail-summary">
                <div className="metric">
                  <div className="k">Log folders (dies)</div>
                  <div className="v">{failAnalysis.n_folders}</div>
                </div>
                <div className="metric">
                  <div className="k">high recommended patterns from pre-process</div>
                  <div className="v">
                    {failAnalysis.summary?.kept_total_fails ?? 0}
                    <span className="v-sub">
                      {" "}
                      across {failAnalysis.summary?.kept_with_fails ?? 0}/
                      {failAnalysis.summary?.kept_n ?? 0} pats
                    </span>
                  </div>
                </div>
                <div className="metric">
                  <div className="k">low risk patterns from pre-process</div>
                  <div className="v">
                    {failAnalysis.summary?.discarded_total_fails ?? 0}
                    <span className="v-sub">
                      {" "}
                      across {failAnalysis.summary?.discarded_with_fails ?? 0}/
                      {failAnalysis.summary?.discarded_n ?? 0} pats
                    </span>
                  </div>
                </div>
                <div className="metric">
                  <div className="k">Files parsed</div>
                  <div className="v">{failAnalysis.files_parsed}</div>
                </div>
              </div>
              <p className="field-hint" style={{ marginBottom: 12 }}>
                Log labels are 1-based (P1…P1000); tables use Verilumen agent
                ids (0…999). Fail = any channel STATUS:F on a die. Channel hits
                and die folders feed criticality ranking.
              </p>

              <NextChipRecommend nextChip={failAnalysis.next_chip} />
            </div>
          )}
        </main>
      </div>
      )}
    </div>
  );
}
