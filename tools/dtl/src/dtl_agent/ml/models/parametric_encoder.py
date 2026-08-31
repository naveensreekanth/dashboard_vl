"""Parametric condition-aware non-sequential ranker."""

from __future__ import annotations

import torch
from torch import nn


class ParametricMLPRanker(nn.Module):
    def __init__(
        self,
        *,
        norm_num_dim: int,
        cand_num_dim: int = 4,
        cond_num_dim: int = 2,
        n_parameter: int = 7,
        n_direction: int = 2,
        n_tight: int = 3,
        n_condition: int = 4,
        n_mode: int = 4,
        embed_dim: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.param_emb = nn.Embedding(n_parameter, embed_dim)
        self.dir_emb = nn.Embedding(n_direction, embed_dim)
        self.tight_emb = nn.Embedding(n_tight, embed_dim)
        self.cond_emb = nn.Embedding(n_condition, embed_dim)
        self.mode_emb = nn.Embedding(n_mode, embed_dim)
        in_dim = norm_num_dim + cand_num_dim + cond_num_dim + 5 * embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        norm_num: torch.Tensor,
        cand_num: torch.Tensor,
        cond_num: torch.Tensor,
        parameter_idx: torch.Tensor,
        direction_idx: torch.Tensor,
        tight_idx: torch.Tensor,
        condition_idx: torch.Tensor,
        mode_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                norm_num,
                cand_num,
                cond_num,
                self.param_emb(parameter_idx),
                self.dir_emb(direction_idx),
                self.tight_emb(tight_idx),
                self.cond_emb(condition_idx),
                self.mode_emb(mode_idx),
            ],
            dim=1,
        )
        return self.net(x).squeeze(1)
