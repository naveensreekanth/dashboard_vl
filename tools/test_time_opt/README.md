# ATE Vector Memory Optimization (React + Node)

Live dashboard for the real test bottleneck:

```text
STIL  --compile-->  Vector memory  --playback-->  DUT pins
```

## Run the live dashboard

```powershell
cd "C:\Users\Mohit\OneDrive\Desktop\vector optimization"
npm run install:all
npm run dev
```

- UI: http://localhost:5173  
- API: http://localhost:8787  

1. Upload a `.stil` file  
2. Click **Run live simulation**  
3. See full-suite vs LSTM vector memory results  

No extra controls — LSTM auto-selects which patterns and how many.

## CLI (optional)

```powershell
python ate_sim.py --keep-ratio 0.6 --period-ns 100
python ate_live_worker.py --stil "C:\Users\Mohit\Downloads\Production_SCAN_stuck_at_1000pat.stil"
```

## Stack

| Piece | Role |
|-------|------|
| `client/` | React + Vite + Recharts live UI |
| `server/` | Node Express upload + SSE |
| `ate_live_worker.py` | Python LSTM / ATE model (JSONL events) |

## Strategies

| Strategy | Meaning |
|----------|---------|
| Full suite | All pattern cycles resident in vector memory |
| LSTM subset | Diversity-selected patterns only (`keep ratio`) |
