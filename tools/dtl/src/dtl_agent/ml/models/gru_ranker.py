"""Core GRU ranker model."""

from __future__ import annotations

import torch
from torch import nn


class CoreGRURanker(nn.Module):
    def __init__(
        self,
        *,
        seq_input_dim: int = 5,
        gru_hidden: int = 64,
        cand_num_dim: int = 4,
        n_parameter: int = 2,
        n_direction: int = 2,
        n_tight: int = 3,
        embed_dim: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.param_emb = nn.Embedding(n_parameter, embed_dim)
        self.dir_emb = nn.Embedding(n_direction, embed_dim)
        self.tight_emb = nn.Embedding(n_tight, embed_dim)
        fusion_in = gru_hidden + cand_num_dim + 3 * embed_dim + 1
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Dropout(dropout),
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        sequence: torch.Tensor,
        cand_num: torch.Tensor,
        parameter_idx: torch.Tensor,
        direction_idx: torch.Tensor,
        tight_idx: torch.Tensor,
        cross_domain: torch.Tensor,
    ) -> torch.Tensor:
        _, h = self.gru(sequence)
        seq_emb = h[-1]
        x = torch.cat(
            [
                seq_emb,
                cand_num,
                self.param_emb(parameter_idx),
                self.dir_emb(direction_idx),
                self.tight_emb(tight_idx),
                cross_domain.unsqueeze(1),
            ],
            dim=1,
        )
        return self.head(x).squeeze(1)
