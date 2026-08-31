"""Training utilities for Phase 7 models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 25
    patience: int = 6
    huber_delta: float = 1.0
    seed: int = 7


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_regressor(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    forward_fn: Callable[[nn.Module, dict], torch.Tensor],
    checkpoint_path: Path,
    config: TrainConfig,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(device)
    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.HuberLoss(delta=config.huber_delta)
    history: list[dict] = []
    best = {"epoch": -1, "val_loss": float("inf")}
    wait = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses = []
        for b in train_loader:
            b = _to_device(b, device)
            pred = forward_fn(model, b)
            loss = loss_fn(pred, b["target"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for b in val_loader:
                b = _to_device(b, device)
                pred = forward_fn(model, b)
                loss = loss_fn(pred, b["target"])
                val_losses.append(float(loss.detach().cpu()))
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
        }
        history.append(rec)
        if rec["val_loss"] < best["val_loss"] - 1e-6:
            best = {"epoch": epoch, "val_loss": rec["val_loss"]}
            torch.save({"model_state": model.state_dict(), "best": best, "config": config.__dict__}, checkpoint_path)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    return best, history


def predict(
    *,
    model: nn.Module,
    loader: DataLoader,
    forward_fn: Callable[[nn.Module, dict], torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval().to(device)
    ys = []
    yp = []
    ids: list[str] = []
    with torch.no_grad():
        for b in loader:
            b = _to_device(b, device)
            pred = forward_fn(model, b).detach().cpu().numpy()
            targ = b["target"].detach().cpu().numpy()
            ys.append(targ)
            yp.append(pred)
            ids.extend(list(b["example_id"]))
    if not ys:
        return np.array([]), np.array([]), []
    return np.concatenate(ys), np.concatenate(yp), ids
