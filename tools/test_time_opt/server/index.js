const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const { spawn } = require("child_process");
const express = require("express");
const cors = require("cors");
const multer = require("multer");

const ROOT = path.resolve(__dirname, "..");
const UPLOAD_DIR = path.join(ROOT, "uploads");
const WORKER = path.join(ROOT, "ate_live_worker.py");
const DEFAULT_STIL = path.join(
  process.env.USERPROFILE || "",
  "Downloads",
  "Production_SCAN_stuck_at_1000pat.stil"
);

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 512 * 1024 * 1024, files: 5000 },
});

const LOG_DIR = path.join(UPLOAD_DIR, "logs");
fs.mkdirSync(LOG_DIR, { recursive: true });

/** @type {Map<string, import('child_process').ChildProcess>} */
const jobs = new Map();

const app = express();
app.use(cors());
app.use(express.json());

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    defaultStilExists: fs.existsSync(DEFAULT_STIL),
  });
});

function resolvePython() {
  return process.env.PYTHON || "python";
}

function runWorker(stilPath, _opts, res) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const jobId = crypto.randomUUID();

  const send = (obj) => {
    res.write(`data: ${JSON.stringify(obj)}\n\n`);
  };

  if (!fs.existsSync(stilPath)) {
    send({ type: "error", message: `STIL not found: ${stilPath}` });
    res.end();
    return;
  }

  const args = [
    WORKER,
    "--stil",
    stilPath,
    "--dropout",
    "0",
    "--budget-mb",
    "0",
    "--min-frac",
    "0.2",
    "--max-frac",
    "0.6",
    "--bits-per-pin",
    "2",
    "--period-ns",
    "100",
    "--max-patterns",
    "0",
    "--refresh-every",
    "25",
  ];

  send({ type: "job", jobId });
  send({ type: "status", message: "Starting simulation worker…" });

  const child = spawn(resolvePython(), args, {
    cwd: ROOT,
    env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1" },
    stdio: ["pipe", "pipe", "pipe"],
  });
  jobs.set(jobId, child);

  let closed = false;
  const endOnce = () => {
    if (closed) return;
    closed = true;
    jobs.delete(jobId);
    try {
      res.end();
    } catch {
      /* ignore */
    }
  };

  res.on("close", () => {
    if (!child.killed) child.kill();
    jobs.delete(jobId);
  });

  let buf = "";
  child.stdout.on("data", (chunk) => {
    buf += chunk.toString("utf8");
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        send(JSON.parse(line));
      } catch {
        send({ type: "log", message: line });
      }
    }
  });

  child.stderr.on("data", (chunk) => {
    const msg = chunk.toString("utf8").trim();
    if (msg) send({ type: "log", message: msg });
  });

  child.on("error", (err) => {
    send({ type: "error", message: err.message });
    endOnce();
  });

  child.on("close", (code) => {
    if (code !== 0 && code !== null) {
      send({ type: "error", message: `Worker exited with code ${code}` });
    }
    endOnce();
  });
}

app.post("/api/seed", (req, res) => {
  const { jobId, pattern_id, auto } = req.body || {};
  if (!jobId) {
    res.status(400).json({ error: "jobId required" });
    return;
  }
  const child = jobs.get(jobId);
  if (!child || !child.stdin || child.killed) {
    res.status(404).json({ error: "Job not waiting for seed (expired or done)" });
    return;
  }
  const payload = auto
    ? { auto: true, pattern_id: pattern_id ?? 0 }
    : { pattern_id: Number(pattern_id) };
  try {
    child.stdin.write(`${JSON.stringify(payload)}\n`);
    res.json({ ok: true, ...payload });
  } catch (err) {
    res.status(500).json({ error: err.message || String(err) });
  }
});

app.post("/api/simulate", upload.single("stil"), (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "Upload a STIL file" });
    return;
  }

  const dest = path.join(
    UPLOAD_DIR,
    `${Date.now()}_${req.file.originalname || "upload.stil"}`
  );
  fs.renameSync(req.file.path, dest);

  runWorker(dest, {}, res);
});

app.get("/api/simulate-default", (_req, res) => {
  runWorker(DEFAULT_STIL, {}, res);
});

/**
 * Upload multiple log folders (files with relative paths) and count
 * FAIL/PASS per pattern for Verilumen kept + discarded lists.
 *
 * multipart fields:
 *   logs[] — files
 *   relative_paths — JSON string array matching files order
 *   selected_ids — JSON array of kept pattern ids
 *   discarded_ids — JSON array of discarded pattern ids
 */
app.post("/api/analyze-fails", upload.array("logs", 5000), (req, res) => {
  try {
    const files = req.files || [];
    if (!files.length) {
      res.status(400).json({ error: "Upload one or more log folders/files" });
      return;
    }

    let relativePaths = [];
    let selectedIds = [];
    let discardedIds = [];
    try {
      relativePaths = JSON.parse(req.body.relative_paths || "[]");
      selectedIds = JSON.parse(req.body.selected_ids || "[]");
      discardedIds = JSON.parse(req.body.discarded_ids || "[]");
    } catch {
      res.status(400).json({ error: "Invalid JSON in relative_paths / selected_ids / discarded_ids" });
      return;
    }

    if (!selectedIds.length && !discardedIds.length) {
      res.status(400).json({
        error: "Run a live simulation first so kept/discarded pattern lists are available",
      });
      return;
    }

    const batchId = `${Date.now()}_${crypto.randomBytes(4).toString("hex")}`;
    const batchDir = path.join(LOG_DIR, batchId);
    fs.mkdirSync(batchDir, { recursive: true });

    const filePairs = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const rel =
        (relativePaths[i] && String(relativePaths[i])) ||
        f.originalname ||
        `file_${i}.log`;
      const safeRel = rel.replace(/\\/g, "/").replace(/^\/+/, "").replace(/\.\./g, "_");
      const dest = path.join(batchDir, safeRel);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.renameSync(f.path, dest);
      filePairs.push({ rel: safeRel, abs: dest });
    }

    const manifest = path.join(batchDir, "_manifest.json");
    fs.writeFileSync(
      manifest,
      JSON.stringify({ files: filePairs.map((p) => p.rel), selectedIds, discardedIds }),
      "utf8"
    );

    const py = resolvePython();
    const script = path.join(ROOT, "log_fail_analyzer.py");
    const args = [
      "-c",
      [
        "import json,sys",
        "from pathlib import Path",
        "from log_fail_analyzer import analyze_uploaded_files",
        "m=json.load(open(sys.argv[1],encoding='utf-8'))",
        "root=Path(sys.argv[2])",
        "pairs=[(r, root/r) for r in m['files']]",
        "out=analyze_uploaded_files(pairs,m['selectedIds'],m['discardedIds'])",
        "print(json.dumps(out))",
      ].join(";"),
      manifest,
      batchDir,
    ];

    const child = spawn(py, args, {
      cwd: ROOT,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (c) => {
      stdout += c.toString("utf8");
    });
    child.stderr.on("data", (c) => {
      stderr += c.toString("utf8");
    });
    child.on("error", (err) => {
      res.status(500).json({ error: err.message || String(err) });
    });
    child.on("close", (code) => {
      if (code !== 0) {
        res.status(500).json({
          error: stderr.trim() || `Analyzer exited with code ${code}`,
        });
        return;
      }
      try {
        const result = JSON.parse(stdout.trim().split("\n").pop());
        res.json(result);
      } catch (err) {
        res.status(500).json({
          error: `Bad analyzer output: ${err.message}`,
          raw: stdout.slice(0, 2000),
        });
      }
    });
  } catch (err) {
    res.status(500).json({ error: err.message || String(err) });
  }
});

// Production: serve built React UI from the same origin as the API.
// Visitors only get the compiled frontend — not your source tree.
const CLIENT_DIST = path.join(ROOT, "client", "dist");
if (fs.existsSync(CLIENT_DIST)) {
  app.use(express.static(CLIENT_DIST, { index: false }));
  app.get("*", (req, res, next) => {
    if (req.path.startsWith("/api")) return next();
    res.sendFile(path.join(CLIENT_DIST, "index.html"), (err) => {
      if (err) next();
    });
  });
}

const PORT = Number(process.env.PORT || 8787);
app.listen(PORT, "0.0.0.0", () => {
  console.log(`ATE vector-memory app on http://0.0.0.0:${PORT}`);
  if (fs.existsSync(CLIENT_DIST)) {
    console.log(`Serving UI from ${CLIENT_DIST}`);
  } else {
    console.log("No client/dist yet — run: npm run build");
  }
});