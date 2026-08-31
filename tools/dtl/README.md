# Dynamic Test Limit (DTL) Recommendation Tool

A semiconductor engineering analysis tool for Dynamic Test Limit (DTL) optimization. It provides 3-month temporal test data analysis, ML-driven limit recommendations (GRU + Hybrid RLS inference), Monte Carlo / parameter simulation, safety and policy validation, and wafer/die-level cost-saving calculations.

---

## 1. Prerequisites & Installation

### Backend Setup (Python 3.10+)

Navigate to the tool directory and install backend dependencies:

```bash
cd tools/dtl
pip install -e .
# or
pip install -r requirements.txt
```

### Frontend Setup (Node.js 18+)

Navigate to the frontend directory and install dependencies:

```bash
cd tools/dtl/frontend
npm install
```

---

## 2. Running the Application

### Start the Backend API Server

From `tools/dtl`:

```bash
python -m uvicorn dtl_agent.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

* API Docs (Swagger): `http://127.0.0.1:8000/docs`
* Health Check: `http://127.0.0.1:8000/api/v1/health`

### Start the Frontend Dev Server

From `tools/dtl/frontend`:

```bash
npm run dev
```

* Dashboard UI: `http://localhost:5173/three-month`

---

## 3. Environment Configuration

Copy `.env.example` to `.env` if local overrides are needed:

```bash
cp .env.example .env
```

Key environment variables:
* `DTL_PORT` (default: `8000`): Backend server port.
* `VITE_API_URL` (default: `http://127.0.0.1:8000`): Backend endpoint for frontend queries.

---

## 4. Runtime Model Assets

The tool uses lightweight, pre-trained PyTorch GRU checkpoints for sequence inference and parameter optimization:

* `artifacts/temporal/shared/checkpoints/core_gru_temporal_v1.pt` (~144 KB)
* `artifacts/temporal/shared/checkpoints/unified_parameter_gru_v1.pt` (~144 KB)
* Metadata and normalization statistics under `artifacts/temporal/shared/`

---

## 5. Three-Month Data Upload Workflow

1. Open the dashboard at `http://localhost:5173/three-month`.
2. In the **Upload Three-Month Test Data** panel, select the 3 monthly measurement CSV files (Month 1, Month 2, Month 3) or a ZIP bundle containing the monthly test records.
3. Click **Process 3-Month Data & Generate Recommendations**.
4. The asynchronous pipeline will ingest the measurements, construct temporal sequences, execute ML GRU ranking and simulation, and render:
   - **Cost Savings Card** (Financial summary, test time reduction, yield impact)
   - **Executive Matrix & Trend Charts** (9-parameter x 3-month grid)
   - **Recommendation & Policy Breakdown** (Current vs Recommended DTL, dynamic validation checks)
   - **Die-Level Analysis** (Lot, Wafer, and Die drilldown)
