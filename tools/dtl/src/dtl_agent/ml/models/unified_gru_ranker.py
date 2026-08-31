"""Unified parameter-aware GRU ranker (Phase 12.5D experiment).

Does not replace CoreGRURanker or ParametricMLPRanker.
Parametric context is fused after the GRU — never tiled into the 200-step sequence.
"""

from __future__ import annotations

import torch
from torch import nn

# Fixed condition order for Option-A context (Phase 12.5C).
PARAMETRIC_CONDITION_ORDER = (
    "COND_RT_NOM",
    "COND_HOT_NOM",
    "COND_RT_LOWV",
    "COND_HOT_HIGHV",
)

# Scorable parameters with candidate + objective_score paths in this repo.
UNIFIED_PARAMETER_VOCAB = (
    "ir_drop",
    "thermal",
    "VMIN",
    "VMAX",
    "IDDQ",
    "SUPPLY_CURRENT",
    "CONTACT_RESISTANCE",
    "INTERCONNECT_RESISTANCE",
    "ON_RESISTANCE",
)

CORE_SCORE_PARAMETERS = frozenset({"ir_drop", "thermal"})
PARAMETRIC_SCORE_PARAMETERS = frozenset(
    {
        "VMIN",
        "VMAX",
        "IDDQ",
        "SUPPLY_CURRENT",
        "CONTACT_RESISTANCE",
        "INTERCONNECT_RESISTANCE",
        "ON_RESISTANCE",
    }
)

# Context = 4 values + 4 availability masks
PARAMETRIC_CONTEXT_DIM = len(PARAMETRIC_CONDITION_ORDER) * 2  # 8


class UnifiedParameterGRURanker(nn.Module):
    """GRU on 200×5 actual-die sequence + post-GRU parameter/context/candidate head."""

    def __init__(
        self,
        *,
        seq_input_dim: int = 5,
        gru_hidden: int = 64,
        cand_num_dim: int = 4,
        parametric_context_dim: int = PARAMETRIC_CONTEXT_DIM,
        n_parameter: int = len(UNIFIED_PARAMETER_VOCAB),
        n_direction: int = 2,
        n_tight: int = 3,
        embed_dim: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_input_dim = seq_input_dim
        self.gru_hidden = gru_hidden
        self.cand_num_dim = cand_num_dim
        self.parametric_context_dim = parametric_context_dim
        self.embed_dim = embed_dim

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
        # has_parametric_context scalar (0/1) concatenated with context vector
        fusion_in = (
            gru_hidden
            + cand_num_dim
            + parametric_context_dim
            + 1
            + 3 * embed_dim
        )
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
        parametric_context: torch.Tensor,
        has_parametric_context: torch.Tensor,
        parameter_idx: torch.Tensor,
        direction_idx: torch.Tensor,
        tight_idx: torch.Tensor,
    ) -> torch.Tensor:
        _, h = self.gru(sequence)
        seq_emb = h[-1]
        if has_parametric_context.dim() == 1:
            has_pc = has_parametric_context.unsqueeze(1)
        else:
            has_pc = has_parametric_context
        x = torch.cat(
            [
                seq_emb,
                cand_num,
                parametric_context,
                has_pc,
                self.param_emb(parameter_idx),
                self.dir_emb(direction_idx),
                self.tight_emb(tight_idx),
            ],
            dim=1,
        )
        return self.head(x).squeeze(1)
