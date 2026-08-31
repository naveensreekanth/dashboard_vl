/**
 * Next-chip recommendation from ATE logs — no caps, no guardrail sampling.
 *
 * 1. RUN (ordered): every pattern with ≥1 fail in logs (greedy order by die coverage)
 * 2. RUN (ML): patterns with no log entry yet, scored ≥ weakest known failer (data-driven cut)
 * 3. SKIP: patterns that only passed in logs (fail=0, pass>0)
 */

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function sigmoid(z) {
  if (z > 20) return 1;
  if (z < -20) return 0;
  return 1 / (1 + Math.exp(-z));
}

function hammingSim(a, b) {
  if (!a?.length || !b?.length) return 0;
  const n = Math.min(a.length, b.length);
  if (!n) return 0;
  let same = 0;
  for (let i = 0; i < n; i++) if (Number(a[i]) === Number(b[i])) same += 1;
  return same / n;
}

function trainLogistic(rows, epochs = 40, lr = 0.25, l2 = 0.01) {
  const dim = rows[0]?.x?.length || 0;
  if (!dim || !rows.length) {
    return { w: [], b: 0, dim: 0 };
  }
  const w = new Array(dim).fill(0);
  let b = 0;

  for (let ep = 0; ep < epochs; ep++) {
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    for (const row of rows) {
      const { x, y } = row;
      let z = b;
      for (let k = 0; k < dim; k++) z += w[k] * x[k];
      const p = sigmoid(z);
      const err = p - y;
      for (let k = 0; k < dim; k++) {
        w[k] -= lr * (err * x[k] + l2 * w[k]);
      }
      b -= lr * err;
    }
  }
  return { w, b, dim };
}

function predictLogistic(model, x) {
  if (!model?.dim || !x?.length) return 0.5;
  let z = model.b;
  for (let k = 0; k < model.dim; k++) z += model.w[k] * (x[k] || 0);
  return sigmoid(z);
}


/**
 * @param {object} analysis
 * @param {object} done
 */
export function recommendNextChipGreedyMl(analysis, done) {
  const selected = new Set((done?.selected_ids || []).map(Number));
  const discarded = new Set((done?.discarded_ids || []).map(Number));
  const allIds = [
    ...new Set([
      ...(done?.selected_ids || []).map(Number),
      ...(done?.discarded_ids || []).map(Number),
    ]),
  ];

  const detailById = {};
  for (const d of done?.selected_details || []) detailById[d.pattern_id] = d;
  for (const d of done?.discarded_details || []) detailById[d.pattern_id] = d;

  const rowById = {};
  for (const r of [...(analysis?.kept || []), ...(analysis?.discarded || [])]) {
    rowById[r.pattern_id] = r;
  }
  for (const u of analysis?.unmatched || []) {
    if (!rowById[u.pattern_id]) {
      rowById[u.pattern_id] = {
        pattern_id: u.pattern_id,
        fails: u.fails || 0,
        passes: u.passes || 0,
        fail_rate:
          (u.fails || 0) / Math.max((u.fails || 0) + (u.passes || 0), 1),
        dies_failed: u.dies_failed || [],
        channel_fails: u.channel_fails || 0,
        recent_fail_mass: 0,
      };
      if (!allIds.includes(u.pattern_id)) allIds.push(u.pattern_id);
    }
  }

  const folders = [...(analysis?.folders || [])].sort();
  const nFolders = Math.max(folders.length, 1);
  const folderIndex = Object.fromEntries(folders.map((f, i) => [f, i]));

  // Recency weight from log folder order (newer folders = later in sorted list)
  const driftWeight = (folder) => {
    const idx = folderIndex[folder];
    if (idx == null || nFolders <= 1) return 1;
    return (idx + 1) / nFolders;
  };

  let maxFails = 1;
  let maxChan = 1;
  let maxDist = 1;
  for (const pid of allIds) {
    const row = rowById[pid];
    if (row?.fails > maxFails) maxFails = row.fails;
    if ((row?.channel_fails || 0) > maxChan) maxChan = row.channel_fails;
    const d = Number(detailById[pid]?.distance_to_nearest);
    if (Number.isFinite(d) && d > maxDist) maxDist = d;
  }

  const failerBits = [];
  for (const pid of allIds) {
    const row = rowById[pid];
    if (!row || row.fails <= 0) continue;
    const bits = detailById[pid]?.bits;
    if (bits?.length) failerBits.push({ pid, bits, fail_rate: row.fail_rate });
  }

  const featureOf = (pid) => {
    const row = rowById[pid] || {
      fails: 0,
      passes: 0,
      fail_rate: 0,
      dies_failed: [],
      channel_fails: 0,
      recent_fail_mass: 0,
    };
    const det = detailById[pid] || {};
    const diesFailed = row.dies_failed?.length || 0;
    const dieFrac = diesFailed / nFolders;
    const chanNorm = (row.channel_fails || 0) / maxChan;
    const discardedFlag = discarded.has(pid) ? 1 : 0;
    const keptFlag = selected.has(pid) ? 1 : 0;
    const dist = Number(det.distance_to_nearest);
    const distNorm = Number.isFinite(dist) ? dist / maxDist : 0.5;
    const recent =
      row.recent_fail_mass ||
      (row.fails || 0) / Math.max((row.fails || 0) + (row.passes || 0), 1);

    let simFail = 0;
    const bits = det.bits;
    if (bits?.length && failerBits.length) {
      let best = 0;
      for (const f of failerBits) {
        if (f.pid === pid) continue;
        best = Math.max(
          best,
          hammingSim(bits, f.bits) * (0.5 + 0.5 * f.fail_rate)
        );
      }
      simFail = best;
    } else if (discarded.has(pid) && det.nearest_kept != null) {
      const nr = rowById[det.nearest_kept];
      if (nr?.fails > 0) {
        simFail = clamp01(1 - (Number.isFinite(dist) ? dist / maxDist : 0.5));
      }
    }

    const x = [
      clamp01(row.fail_rate || 0),
      clamp01(Math.log1p(row.fails || 0) / Math.log1p(maxFails)),
      clamp01(dieFrac),
      clamp01(chanNorm),
      discardedFlag,
      keptFlag,
      clamp01(distNorm),
      clamp01(simFail),
      clamp01(recent),
    ];
    return { x, simFail, dieFrac, chanNorm, recent, discardedFlag, keptFlag };
  };

  const trainRows = [];
  for (const pid of allIds) {
    const row = rowById[pid];
    if (!row) continue;
    const total = (row.fails || 0) + (row.passes || 0);
    if (!total) continue;
    const { x } = featureOf(pid);
    trainRows.push({ x, y: row.fails > 0 ? 1 : 0, pid });
  }

  const model = trainLogistic(trainRows.map((r) => ({ x: r.x, y: r.y })));

  const scored = allIds.map((pid) => {
    const row = rowById[pid] || {
      pattern_id: pid,
      fails: 0,
      passes: 0,
      fail_rate: 0,
      dies_failed: [],
      channel_fails: 0,
    };
    const total = (row.fails || 0) + (row.passes || 0);
    const feat = featureOf(pid);
    const mlRisk = predictLogistic(model, feat.x);
    const coverSet = new Set(row.dies_failed || []);
    let coverMass = 0;
    for (const d of coverSet) coverMass += driftWeight(d);

    return {
      pattern_id: pid,
      fails: row.fails || 0,
      passes: row.passes || 0,
      total,
      fail_rate: row.fail_rate || 0,
      channel_fails: row.channel_fails || 0,
      dies_failed: [...coverSet],
      coverSet,
      coverMass,
      mlRisk,
      simFail: feat.simFail,
      kept: selected.has(pid),
      discarded: discarded.has(pid),
      in_logs: total > 0,
      log_pass_only: total > 0 && row.fails === 0,
      not_in_logs: total === 0,
    };
  });

  const failers = scored.filter((s) => s.fails > 0);
  const nFailers = failers.length;

  const universe = new Set();
  for (const s of failers) {
    for (const d of s.coverSet) universe.add(d);
  }

  const failerMlRisks = failers.map((s) => s.mlRisk);
  const failerSimScores = failers.map((s) => s.simFail).filter((v) => v > 0);
  const mlCut =
    failerMlRisks.length > 0 ? Math.min(...failerMlRisks) : Infinity;
  const simCut =
    failerSimScores.length > 0 ? Math.min(...failerSimScores) : Infinity;

  const picked = [];
  const pickedSet = new Set();

  const pushPick = (s, tier, score, reasons, diesCoveredNew = 0) => {
    if (pickedSet.has(s.pattern_id)) return;
    pickedSet.add(s.pattern_id);
    picked.push({
      pattern_id: s.pattern_id,
      tier,
      rank: picked.length + 1,
      score: Number(score.toFixed(4)),
      ml_risk: Number(s.mlRisk.toFixed(4)),
      fails: s.fails,
      passes: s.passes,
      fail_rate: s.fail_rate,
      dies_covered_new: diesCoveredNew,
      kept: s.kept,
      discarded: s.discarded,
      reasons,
    });
  };

  const singleDie = nFolders === 1;

  // --- 1) All log failers ---
  // One log = one die/chip: rank by severity (channels, fail rate, fails).
  // Multiple logs = multiple dies: greedy order by die coverage first.
  if (singleDie) {
    const ranked = [...failers].sort(
      (a, b) =>
        b.channel_fails - a.channel_fails ||
        b.fail_rate - a.fail_rate ||
        b.fails - a.fails ||
        b.mlRisk - a.mlRisk ||
        a.pattern_id - b.pattern_id
    );
    for (const s of ranked) {
      const score =
        s.channel_fails * 10 + s.fail_rate * 5 + Math.log1p(s.fails) + s.mlRisk;
      const reasons = [
        "failed on this chip (1 log = 1 die)",
        `${s.fails} fail(s), ${(s.fail_rate * 100).toFixed(1)}% fail rate`,
      ];
      if (s.discarded) reasons.push("pre-process skip, failed on silicon");
      if (s.channel_fails > 0) {
        reasons.push(`${s.channel_fails} channel STATUS:F`);
      }
      reasons.push(`ML risk ${(s.mlRisk * 100).toFixed(0)}%`);
      pushPick(s, "log_fail", score, reasons);
    }
  } else {
    const uncovered = new Set(universe);
    const orderedFailers = [];

    while (orderedFailers.length < failers.length) {
      let best = null;
      let bestScore = -1;
      for (const s of failers) {
        if (pickedSet.has(s.pattern_id) || orderedFailers.includes(s)) continue;
        let newMass = 0;
        let newCount = 0;
        for (const d of s.coverSet) {
          if (!uncovered.has(d)) continue;
          newCount += 1;
          newMass += driftWeight(d);
        }
        const score =
          newCount > 0
            ? newMass * (1 + s.fail_rate) * Math.log1p(s.fails) * (1 + s.mlRisk)
            : s.fail_rate * Math.log1p(s.fails) * (1 + s.mlRisk);
        if (score > bestScore) {
          bestScore = score;
          best = { s, newCount, score };
        }
      }
      if (!best) break;
      orderedFailers.push(best.s);
      for (const d of best.s.coverSet) uncovered.delete(d);

      const reasons = ["failed in ATE logs"];
      if (best.newCount > 0) {
        reasons.push(`covers ${best.newCount} failing die(s) not yet in order`);
      } else {
        reasons.push("failed in logs (die already covered by earlier pick)");
      }
      reasons.push(
        `${best.s.fails} fail(s), ${(best.s.fail_rate * 100).toFixed(1)}% fail rate`
      );
      if (best.s.discarded) reasons.push("pre-process skip, failed on silicon");
      if (best.s.channel_fails > 0) {
        reasons.push(`${best.s.channel_fails} channel STATUS:F`);
      }
      reasons.push(`ML risk ${(best.s.mlRisk * 100).toFixed(0)}%`);

      pushPick(
        best.s,
        best.newCount > 0 ? "log_fail_greedy" : "log_fail",
        best.score,
        reasons,
        best.newCount
      );
    }

    for (const s of failers) {
      if (!pickedSet.has(s.pattern_id)) {
        pushPick(s, "log_fail", s.fail_rate, [
          "failed in ATE logs",
          `${s.fails} fail(s)`,
        ]);
      }
    }
  }

  const covered = singleDie ? (nFailers > 0 ? 1 : 0) : universe.size;

  // --- 2) ML: patterns NOT in logs — include if score ≥ weakest known failer ---
  if (nFailers > 0) {
    const mlCandidates = scored
      .filter((s) => !pickedSet.has(s.pattern_id))
      .filter((s) => s.not_in_logs)
      .map((s) => ({
        s,
        hunt: Math.max(
          s.mlRisk >= mlCut ? s.mlRisk : 0,
          s.simFail >= simCut ? s.simFail : 0
        ),
        viaMl: s.mlRisk >= mlCut,
        viaSim: s.simFail >= simCut,
      }))
      .filter((h) => h.viaMl || h.viaSim)
      .sort((a, b) => b.hunt - a.hunt || b.s.mlRisk - a.s.mlRisk);

    for (const h of mlCandidates) {
      const reasons = ["not seen in logs — ML / similarity vs known failers"];
      if (h.viaMl) {
        reasons.push(
          `ML risk ${(h.s.mlRisk * 100).toFixed(0)}% (cut ${(mlCut * 100).toFixed(0)}% from failers)`
        );
      }
      if (h.viaSim) {
        reasons.push(
          `similar to failers ${(h.s.simFail * 100).toFixed(0)}% (cut ${(simCut * 100).toFixed(0)}%)`
        );
      }
      if (h.s.discarded) reasons.push("pre-process skip list");
      pushPick(h.s, "ml_unseen", h.hunt, reasons);
    }
  }

  // --- 3) Log pass-only → skip (explicit in skip list) ---
  const skipIds = allIds.filter((pid) => !pickedSet.has(pid));
  const skipPassOnly = scored.filter((s) => s.log_pass_only && !pickedSet.has(s.pattern_id));
  const skipNotInLogs = scored.filter((s) => s.not_in_logs && !pickedSet.has(s.pattern_id));

  picked.forEach((p, i) => {
    p.rank = i + 1;
  });

  return {
    algorithm: singleDie
      ? "single-die: all log failers ranked by severity; log pass-only skipped"
      : "multi-die: all log failers (greedy die order) + ML for not-in-log",
    n_dies: nFolders,
    single_die_mode: singleDie,
    n_patterns_total: allIds.length,
    n_failing_patterns: nFailers,
    n_log_pass_skip: skipPassOnly.length,
    n_not_in_logs_skip: skipNotInLogs.length,
    n_failing_dies: universe.size,
    n_dies_covered: covered,
    coverage_pct: universe.size
      ? Math.round((100 * covered) / universe.size)
      : 100,
    n_log_fail: picked.filter(
      (p) => p.tier === "log_fail_greedy" || p.tier === "log_fail"
    ).length,
    n_ml_unseen: picked.filter((p) => p.tier === "ml_unseen").length,
    n_skip_for_next: skipIds.length,
    recommend: picked,
    skip_ids_head: skipIds.slice(0, 100),
    thresholds: {
      ml_cut_from_failers: nFailers ? mlCut : null,
      sim_cut_from_failers: nFailers ? simCut : null,
    },
    model: {
      type: "logistic_regression_sgd",
      features: [
        "fail_rate",
        "log_fails_norm",
        "die_fail_frac",
        "channel_norm",
        "discarded",
        "kept",
        "embed_dist_norm",
        "sim_to_failers",
        "recent_fail_mass",
      ],
      train_n: trainRows.length,
      train_pos: trainRows.filter((r) => r.y === 1).length,
    },
  };
}
