"""Joint fusion model with optional Core branch mask."""

from __future__ import annotations

import torch
from torch import nn

from dtl_agent.ml.models.gru_ranker import CoreGRURanker
from dtl_agent.ml.models.parametric_encoder import ParametricMLPRanker


class JointRanker(nn.Module):
    """Small joint model for linked + parametric-only routing.

    For core-domain rows, provide real `sequence` and core categorical features.
    For parametric-only rows, set `has_core=0` and pass zero sequence tensor.
    """

    def __init__(self, *, norm_num_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.core = CoreGRURanker(dropout=dropout)
        self.param = ParametricMLPRanker(norm_num_dim=norm_num_dim, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        has_core = batch["has_core"]
        core_score = torch.zeros_like(has_core)
        core_idx = torch.where(has_core > 0.5)[0]
        if core_idx.numel() > 0:
            core_score_sub = self.core(
                sequence=batch["sequence"][core_idx],
                cand_num=batch["cand_num"][core_idx],
                parameter_idx=batch["parameter_idx"][core_idx],
                direction_idx=batch["direction_idx"][core_idx],
                tight_idx=batch["tight_idx"][core_idx],
                cross_domain=batch["cross_domain"][core_idx],
            )
            core_score[core_idx] = core_score_sub
        param_score = self.param(
            norm_num=batch["norm_num"],
            cand_num=batch["cand_num"],
            cond_num=batch["cond_num"],
            parameter_idx=batch["parameter_idx"],
            direction_idx=batch["direction_idx"],
            tight_idx=batch["tight_idx"],
            condition_idx=batch["condition_idx"],
            mode_idx=batch["mode_idx"],
        )
        x = torch.stack([core_score * has_core, param_score], dim=1)
        return self.fuse(x).squeeze(1)
