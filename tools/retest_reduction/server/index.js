require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const PORT = process.env.PORT || 3001;
const FASTAPI_BASE = process.env.FASTAPI_BASE || 'http://127.0.0.1:8000';

// Multer: store uploads in memory
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '50mb' }));

// ─────────────────────────────────────────
// Helper: forward a simple GET to FastAPI
// ─────────────────────────────────────────
async function forwardGet(fastapiPath, req, res) {
  try {
    const params = req.query || {};
    const r = await axios.get(`${FASTAPI_BASE}${fastapiPath}`, { params });
    res.json(r.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json(err.response?.data || { detail: err.message });
  }
}

// ─────────────────────────────────────────
// Helper: forward a simple POST to FastAPI
// ─────────────────────────────────────────
async function forwardPost(fastapiPath, req, res) {
  try {
    const r = await axios.post(`${FASTAPI_BASE}${fastapiPath}`, req.body);
    res.json(r.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json(err.response?.data || { detail: err.message });
  }
}

// ─────────────────────────────────────────
// HEALTH
// ─────────────────────────────────────────
app.get('/api/health', (req, res) => forwardGet('/health', req, res));

// ─────────────────────────────────────────
// MODEL INFO
// ─────────────────────────────────────────
app.get('/api/model/info', (req, res) => forwardGet('/model/info', req, res));

// ─────────────────────────────────────────
// SINGLE EVENT OPTIONS
// ─────────────────────────────────────────
app.get('/api/datasets/single-event-options', (req, res) =>
  forwardGet('/datasets/single-event-options', req, res)
);

// ─────────────────────────────────────────
// PREDICT SINGLE
// ─────────────────────────────────────────
app.post('/api/predict', (req, res) => forwardPost('/predict', req, res));
app.post('/api/predict/single-with-shap', (req, res) =>
  forwardPost('/predict/single-with-shap', req, res)
);

// ─────────────────────────────────────────
// PREDICT BATCH
// ─────────────────────────────────────────
app.post('/api/predict/batch', (req, res) =>
  forwardPost('/predict/batch', req, res)
);

// ─────────────────────────────────────────
// MONTH 12 BATCH (scored from built-in file)
// ─────────────────────────────────────────
app.get('/api/analysis/month12-batch', (req, res) =>
  forwardGet('/analysis/month12-batch', req, res)
);

// ─────────────────────────────────────────
// UPLOAD PRE-RETEST XLSX
// ─────────────────────────────────────────
app.post('/api/analysis/upload-pre-retest', upload.single('file'), async (req, res) => {
  try {
    const fd = new FormData();
    if (req.file) {
      fd.append('file', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });
    }
    const costPerHour = req.body.cost_per_hour || req.query.cost_per_hour || '';
    if (costPerHour) fd.append('cost_per_hour', costPerHour);

    const r = await axios.post(`${FASTAPI_BASE}/analysis/upload-pre-retest`, fd, {
      headers: fd.getHeaders(),
    });
    res.json(r.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json(err.response?.data || { detail: err.message });
  }
});

// ─────────────────────────────────────────
// VALIDATE OUTCOMES
// ─────────────────────────────────────────
app.post('/api/analysis/validate-outcomes', upload.single('file'), async (req, res) => {
  try {
    const fd = new FormData();
    if (req.file) {
      fd.append('file', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });
    }
    const useLocal = req.body.use_local_file || 'False';
    const predictions = req.body.predictions || '[]';
    fd.append('use_local_file', useLocal);
    fd.append('predictions', predictions);

    const r = await axios.post(`${FASTAPI_BASE}/analysis/validate-outcomes`, fd, {
      headers: fd.getHeaders(),
    });
    res.json(r.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json(err.response?.data || { detail: err.message });
  }
});

// ─────────────────────────────────────────
// HISTORICAL VALIDATION
// ─────────────────────────────────────────
app.get('/api/analysis/historical-validation', (req, res) =>
  forwardGet('/analysis/historical-validation', req, res)
);

// ─────────────────────────────────────────
// REFERENCE AUDIT
// ─────────────────────────────────────────
app.get('/api/analysis/reference-audit', (req, res) =>
  forwardGet('/analysis/reference-audit', req, res)
);

// ─────────────────────────────────────────
// ONLINE LEARNING
// ─────────────────────────────────────────
app.get('/api/online-learning/status', (req, res) =>
  forwardGet('/online-learning/status', req, res)
);
app.post('/api/online-learning/learn', (req, res) =>
  forwardPost('/online-learning/learn', req, res)
);
app.post('/api/online-learning/reset', (req, res) =>
  forwardPost('/online-learning/reset', req, res)
);

// ─────────────────────────────────────────
// COST IMPACT
// ─────────────────────────────────────────
app.post('/api/cost-impact', (req, res) => forwardPost('/cost-impact', req, res));

// ─────────────────────────────────────────
// START
// ─────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n  ⚡ ATE Retest AI — Node.js API Bridge`);
  console.log(`  → Express:  http://localhost:${PORT}`);
  console.log(`  → FastAPI:  ${FASTAPI_BASE}\n`);
});
