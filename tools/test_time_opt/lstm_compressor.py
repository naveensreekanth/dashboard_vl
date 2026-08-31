"""
LSTM that compresses a scan-pattern sequence into a fixed-size state.

Supports neural-net Dropout (randomly zeros hidden units during forward).
Dropout regularizes embeddings; it does not by itself shrink ATE vector memory.
ATE RAM reduction still comes from pattern subset selection (pattern keep ratio).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatternLSTMCompressor(nn.Module):
    """
    Maps (batch, seq_len, n_channels) scan bits → compact embedding.

    Only the last hidden state is retained as the pattern embedding.
    """

    def __init__(
        self,
        n_channels: int = 23,
        hidden_size: int = 64,
        num_layers: int = 2,
        embed_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dropout_p = float(dropout)

        self.input_proj = nn.Linear(n_channels, hidden_size)
        self.input_dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_p if num_layers > 1 else 0.0,
        )
        self.hidden_dropout = nn.Dropout(self.dropout_p)
        self.out_proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float tensor (B, T, C)
        returns: (B, embed_dim)
        """
        h = torch.tanh(self.input_proj(x))
        h = self.input_dropout(h)
        _, (h_n, _) = self.lstm(h)
        last = self.hidden_dropout(h_n[-1])
        return self.out_proj(last)

    def state_nbytes(self, dtype=torch.float32) -> int:
        bpe = torch.tensor([], dtype=dtype).element_size()
        h_c = 2 * self.num_layers * self.hidden_size * bpe
        emb = self.embed_dim * bpe
        return h_c + emb

    def param_nbytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())
