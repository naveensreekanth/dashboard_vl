"""Pure NumPy Recursive Least Squares regressor (experimental)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RLSState:
    n_features: int
    forgetting_factor: float
    delta: float
    n_updates: int
    theta: np.ndarray
    P: np.ndarray


class RLSRegressor:
    """Standard RLS with forgetting factor λ and initial covariance δ^{-1} I.

    Predicts a scalar score y ≈ θᵀ x. Designed for online updates after an
    optional offline warm-start via :meth:`fit`.
    """

    def __init__(
        self,
        n_features: int,
        *,
        forgetting_factor: float = 0.995,
        delta: float = 10.0,
        theta: np.ndarray | None = None,
        P: np.ndarray | None = None,
        n_updates: int = 0,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must be in (0, 1]")
        if delta <= 0.0:
            raise ValueError("delta must be > 0")
        self.n_features = int(n_features)
        self.forgetting_factor = float(forgetting_factor)
        self.delta = float(delta)
        self.n_updates = int(n_updates)
        if theta is None:
            self.theta = np.zeros(self.n_features, dtype=np.float64)
        else:
            self.theta = np.asarray(theta, dtype=np.float64).reshape(self.n_features).copy()
        if P is None:
            self.P = (1.0 / self.delta) * np.eye(self.n_features, dtype=np.float64)
        else:
            self.P = np.asarray(P, dtype=np.float64).reshape(self.n_features, self.n_features).copy()

    def predict_one(self, x: np.ndarray) -> float:
        xv = np.asarray(x, dtype=np.float64).reshape(self.n_features)
        return float(self.theta @ xv)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xm = np.asarray(X, dtype=np.float64)
        if Xm.ndim == 1:
            Xm = Xm.reshape(1, -1)
        if Xm.shape[1] != self.n_features:
            raise ValueError(f"X has {Xm.shape[1]} features, expected {self.n_features}")
        return Xm @ self.theta

    def update(self, x: np.ndarray, y: float) -> float:
        """Incorporate one observation; return absolute residual before update."""
        xv = np.asarray(x, dtype=np.float64).reshape(self.n_features)
        yv = float(y)
        lam = self.forgetting_factor
        err = yv - float(self.theta @ xv)
        Px = self.P @ xv
        denom = lam + float(xv @ Px)
        if abs(denom) < 1e-18:
            # Ill-conditioned; skip covariance update but still nudge theta lightly.
            self.theta = self.theta + (1e-6 * err) * xv
            self.n_updates += 1
            return abs(err)
        g = Px / denom
        self.theta = self.theta + g * err
        # Joseph-form style update, then stabilize for long λ=1 streams.
        P_new = (self.P - np.outer(g, Px)) / lam
        self.P = 0.5 * (P_new + P_new.T)
        self._stabilize()
        self.n_updates += 1
        return abs(err)

    def _stabilize(self) -> None:
        """Keep θ/P finite and bound covariance growth during long online streams."""
        if not np.isfinite(self.theta).all():
            self.theta = np.nan_to_num(self.theta, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(self.P).all():
            self.P = (1.0 / self.delta) * np.eye(self.n_features, dtype=np.float64)
            return
        self.P = 0.5 * (self.P + self.P.T)
        max_diag = max(1e4 / self.delta, 1.0)
        d = np.diag(self.P)
        peak = float(np.max(d))
        if peak > max_diag:
            self.P *= max_diag / peak

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        reset: bool = True,
    ) -> "RLSRegressor":
        """Warm-start by sequential RLS updates (deterministic row order)."""
        Xm = np.asarray(X, dtype=np.float64)
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        if Xm.ndim != 2:
            raise ValueError("X must be 2-D")
        if Xm.shape[0] != yv.shape[0]:
            raise ValueError("X/y length mismatch")
        if Xm.shape[1] != self.n_features:
            raise ValueError(f"X has {Xm.shape[1]} features, expected {self.n_features}")
        if reset:
            self.theta = np.zeros(self.n_features, dtype=np.float64)
            self.P = (1.0 / self.delta) * np.eye(self.n_features, dtype=np.float64)
            self.n_updates = 0
        for i in range(Xm.shape[0]):
            self.update(Xm[i], float(yv[i]))
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "forgetting_factor": self.forgetting_factor,
            "delta": self.delta,
            "n_updates": self.n_updates,
            "theta": self.theta.tolist(),
            "P": self.P.tolist(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> "RLSRegressor":
        self.n_features = int(state["n_features"])
        self.forgetting_factor = float(state["forgetting_factor"])
        self.delta = float(state["delta"])
        self.n_updates = int(state["n_updates"])
        self.theta = np.asarray(state["theta"], dtype=np.float64).reshape(self.n_features)
        self.P = np.asarray(state["P"], dtype=np.float64).reshape(self.n_features, self.n_features)
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "dtl_rls_v1", **self.state_dict()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RLSRegressor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            n_features=int(payload["n_features"]),
            forgetting_factor=float(payload["forgetting_factor"]),
            delta=float(payload["delta"]),
            n_updates=int(payload.get("n_updates", 0)),
            theta=np.asarray(payload["theta"], dtype=np.float64),
            P=np.asarray(payload["P"], dtype=np.float64),
        )
        return obj

    def model_nbytes(self) -> int:
        return int(self.theta.nbytes + self.P.nbytes)
