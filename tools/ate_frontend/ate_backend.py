from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime
import random

app = FastAPI(title="ATE Intelligence Local Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock Data Generators
def generate_die_grid(wafer_id: str, total_dies: int = 196):
    grid = []
    size = int(total_dies ** 0.5)
    for row in range(size):
        for col in range(size):
            # Circular wafer die filter
            center = size / 2
            dist = ((row - center) ** 2 + (col - center) ** 2) ** 0.5
            if dist > center:
                continue
            
            rand = random.random()
            if rand > 0.15:
                res = "pass"
            elif rand > 0.08:
                res = "retest"
            elif rand > 0.03:
                res = "fail"
            else:
                res = "reclass"
                
            grid.append({
                "die_id": f"DIE-{row:02d}-{col:02d}",
                "wafer_id": wafer_id,
                "x": col,
                "y": row,
                "row": row,
                "column": col,
                "result": res,
                "bin": res,
                "fail_code": "FREQ_MARGIN" if res == "fail" else None,
                "test_time_ms": random.randint(45, 120),
                "confidence": round(random.uniform(0.85, 0.99), 2),
                "timestamp": datetime.datetime.now().isoformat()
            })
    return grid

MOCK_WAFERS = {
    "WFR-9082": {
        "wafer_id": "WFR-9082",
        "lot_id": "LOT-2026-A1",
        "status": "COMPLETED",
        "yield_pct": 94.2,
        "total_dies": 180,
        "tested_dies": 180,
        "caption": "Lot 2026-A1 - Wafer 12 (300mm Silicon)",
        "bin_counts": {"pass": 169, "retest": 6, "fail": 4, "reclass": 1},
        "pass_count": 169,
        "fail_count": 4,
        "retest_count": 6,
        "reclass_count": 1,
        "updated_at": datetime.datetime.now().isoformat()
    }
}

MOCK_DIES = {
    "WFR-9082": generate_die_grid("WFR-9082", 225)
}

MOCK_KPIS = [
    {
        "id": "yield_improvement",
        "name": "Yield Improvement",
        "title": "Yield Improvement",
        "value": 3.8,
        "unit": "%",
        "baseline": 91.2,
        "target": 95.0,
        "previous_value": 3.2,
        "improvement": 0.6,
        "trend": "up",
        "status": "OPTIMAL",
        "spark": [91.2, 92.0, 92.5, 93.1, 93.8, 94.2, 95.0],
        "series": [91.2, 92.0, 92.5, 93.1, 93.8, 94.2, 95.0],
        "description": "AI adaptive limit optimization yield gain"
    },
    {
        "id": "test_time_reduction",
        "name": "Test Time Savings",
        "title": "Test Time Savings",
        "value": 18.5,
        "unit": "%",
        "baseline": 100,
        "target": 80,
        "previous_value": 16.2,
        "improvement": 2.3,
        "trend": "up",
        "status": "OPTIMAL",
        "spark": [100, 94, 90, 86, 84, 82.5, 81.5],
        "series": [100, 94, 90, 86, 84, 82.5, 81.5],
        "description": "LSTM pattern selection test cycle speedup"
    },
    {
        "id": "false_failure_reduction",
        "name": "False Failure Reduction",
        "title": "False Failure Reduction",
        "value": 42.1,
        "unit": "%",
        "baseline": 0,
        "target": 50,
        "previous_value": 38.0,
        "improvement": 4.1,
        "trend": "up",
        "status": "GOOD",
        "spark": [10, 18, 25, 30, 36, 40, 42.1],
        "series": [10, 18, 25, 30, 36, 40, 42.1],
        "description": "Dynamic guardband Cpk tuning escape prevention"
    }
]

MOCK_LIMITS = [
    {
        "limit_id": "LIM-VDD-01",
        "parameter": "VDD_MIN_V",
        "test_name": "Low-VDD Functional Speedpath",
        "name": "VDD Core Minimum Voltage",
        "site_id": "SITE-01",
        "tester_id": "ADV-93K-01",
        "lot_id": "LOT-2026-A1",
        "previous_limit": 0.75,
        "current_limit": 0.72,
        "delta": -0.03,
        "change_percentage": -4.0,
        "change_pct": -4.0,
        "change_label": "-0.03 V (Tightened)",
        "direction": "tightened",
        "cpk": 1.67,
        "target_cpk": 1.50,
        "confidence": 0.98,
        "reason": "Process capability Cpk = 1.67 exceeds target 1.50. Tightened limit to prevent marginal timing escapes.",
        "status": "RECOMMENDED",
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat()
    },
    {
        "limit_id": "LIM-FREQ-02",
        "parameter": "FMAX_GHZ",
        "test_name": "MBIST Fmax Characterization",
        "name": "Maximum Memory Frequency",
        "site_id": "SITE-02",
        "tester_id": "ADV-93K-02",
        "lot_id": "LOT-2026-A1",
        "previous_limit": 2.10,
        "current_limit": 2.25,
        "delta": 0.15,
        "change_percentage": 7.1,
        "change_pct": 7.1,
        "change_label": "+0.15 GHz (Widened)",
        "direction": "widened",
        "cpk": 1.82,
        "target_cpk": 1.50,
        "confidence": 0.95,
        "reason": "RANSAC Shmoo boundary supports higher frequency binning with 99.2% confidence.",
        "status": "ACTIVE",
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat()
    }
]

MOCK_EVENTS = [
    {
        "event_id": "EVT-1001",
        "event_type": "LIMIT_RECOMMENDATION",
        "timestamp": datetime.datetime.now().isoformat(),
        "tag": "info",
        "text": "Dynamic limit LIM-VDD-01 recommended tightening based on 3-month Cpk trend.",
        "lot_id": "LOT-2026-A1",
        "wafer_id": "WFR-9082",
        "tester_id": "ADV-93K-01"
    },
    {
        "event_id": "EVT-1002",
        "event_type": "SHMOO_OPTIMIZATION",
        "timestamp": datetime.datetime.now().isoformat(),
        "tag": "pass",
        "text": "Shmoo ML classifier verified Normal Pass region for Wafer WFR-9082.",
        "lot_id": "LOT-2026-A1",
        "wafer_id": "WFR-9082",
        "tester_id": "ADV-93K-02"
    }
]

MOCK_MAINTENANCE = {
    "flagged_count": 1,
    "model_available": True,
    "assets": [
        {
            "asset_id": "AST-ADV-93K-01",
            "name": "Advantest V93000 Tester #1 Pin Electronics",
            "health_pct": 92.5,
            "status": "HEALTHY",
            "rul_days": 145,
            "tester_id": "ADV-93K-01",
            "component": "PE_CARD_3",
            "failure_probability": 0.04,
            "confidence": 0.96,
            "severity": "LOW",
            "recommended_action": "Routine calibration at next planned maintenance cycle.",
            "model_available": True,
            "updated_at": datetime.datetime.now().isoformat()
        }
    ]
}

# Routes
@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "database": True, "redis": True}

@app.get("/ready")
@app.get("/api/ready")
def ready():
    return {"status": "ready", "database": True, "redis": True, "websocket_clients": 1}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/login")
@app.post("/api/auth/login")
def login(req: LoginRequest):
    return {
        "access_token": "local-verilumen-jwt-token",
        "token_type": "bearer",
        "role": req.username if req.username in ["viewer", "admin", "test_eng", "process_eng", "ai_eng", "maint_eng"] else "viewer",
        "username": req.username,
        "user_id": "USR-LOCAL-01",
        "expires_in_minutes": 1440
    }

@app.get("/auth/me")
@app.get("/api/auth/me")
def auth_me():
    return {
        "user_id": "USR-LOCAL-01",
        "username": "engineer",
        "full_name": "ATE Test Engineer",
        "role": "test_eng",
        "permissions": ["*"]
    }

@app.get("/dashboard/summary")
@app.get("/api/dashboard/summary")
def dashboard_summary():
    return {
        "header": {
            "lots_in_test": 4,
            "test_time_saved_hours": 142.5,
            "overall_yield_pct": 94.2
        },
        "active_wafer": MOCK_WAFERS["WFR-9082"],
        "kpis": MOCK_KPIS,
        "maintenance": MOCK_MAINTENANCE,
        "test_limits": {
            "adjustments_today": len(MOCK_LIMITS),
            "items": MOCK_LIMITS
        },
        "recent_events": MOCK_EVENTS,
        "connection_hint": "Connected to local ATE Intelligence Backend (Offline Mode)"
    }

@app.get("/wafers/{wafer_id}")
@app.get("/api/wafers/{wafer_id}")
def get_wafer(wafer_id: str):
    return MOCK_WAFERS.get(wafer_id, MOCK_WAFERS["WFR-9082"])

@app.get("/wafers/{wafer_id}/dies")
@app.get("/api/wafers/{wafer_id}/dies")
def get_wafer_dies(wafer_id: str):
    return MOCK_DIES.get(wafer_id, MOCK_DIES["WFR-9082"])

@app.get("/kpis")
@app.get("/api/kpis")
def get_kpis():
    return {"kpis": MOCK_KPIS}

@app.get("/kpis/{kpi_id}")
@app.get("/api/kpis/{kpi_id}")
def get_kpi(kpi_id: str):
    for k in MOCK_KPIS:
        if k["id"] == kpi_id:
            return k
    return MOCK_KPIS[0]

@app.get("/kpis/{kpi_id}/history")
@app.get("/api/kpis/{kpi_id}/history")
def get_kpi_history(kpi_id: str):
    return {
        "kpi_id": kpi_id,
        "history": [
            {"t": (datetime.datetime.now() - datetime.timedelta(hours=i)).isoformat(), "v": round(90 + i*0.2, 2)}
            for i in range(24, 0, -1)
        ]
    }

@app.get("/events")
@app.get("/api/events")
def get_events():
    return {"events": MOCK_EVENTS, "total": len(MOCK_EVENTS)}

@app.get("/maintenance")
@app.get("/api/maintenance")
def get_maintenance():
    return MOCK_MAINTENANCE

@app.get("/test-limits")
@app.get("/api/test-limits")
def get_test_limits():
    return {"adjustments_today": len(MOCK_LIMITS), "items": MOCK_LIMITS}

@app.post("/test-limits/{limit_id}/approve")
@app.post("/api/test-limits/{limit_id}/approve")
def approve_limit(limit_id: str):
    for l in MOCK_LIMITS:
        if l["limit_id"] == limit_id:
            l["status"] = "ACTIVE"
            return l
    return MOCK_LIMITS[0]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
