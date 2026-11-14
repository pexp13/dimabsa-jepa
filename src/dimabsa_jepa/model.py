"""Encoder + VA regression head + I-JEPA-style latent predictor."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def sinusoidal_positions(max_len: int, dim: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class JepaPredictor(nn.Module):
    """Narrow transformer: context latents + mask tokens -> target latents."""

    def __init__(self, hidden: int, dim: int = 256, layers: int = 2, heads: int = 4, max_len: int = 512):
        super().__init__()
        self.proj_in = nn.Linear(hidden, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sinusoidal_positions(max_len, dim), persistent=False)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, hidden)

    def forward(self, ctx: torch.Tensor, attention_mask: torch.Tensor, jepa_mask: torch.Tensor) -> torch.Tensor:
        z = self.proj_in(ctx)
        z = torch.where(jepa_mask.unsqueeze(-1), self.mask_token.to(z.dtype), z)
        z = z + self.pos[: z.size(1)].to(z.dtype)
        z = self.blocks(z, src_key_padding_mask=~attention_mask.bool())
        return self.proj_out(self.norm(z))


class DimAsrJepa(nn.Module):
    """Context encoder (trained), EMA target encoder, VA head and JEPA predictor.

    The JEPA objective follows I-JEPA: tokens under `jepa_mask` are removed from
    the context encoder's view (attention-masked, so nothing can attend to
    them), and the predictor must regress the target encoder's representation
    of those tokens, computed from the full unmasked input. Nothing is
    reconstructed in token space.
    """

    def __init__(self, model_name: str, use_jepa: bool = True, pred_dim: int = 256,
                 pred_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden, 2),
        )
        self.use_jepa = use_jepa
        if use_jepa:
            self.target = copy.deepcopy(self.encoder)
            self.target.requires_grad_(False)
            self.target.eval()
            self.predictor = JepaPredictor(hidden, pred_dim, pred_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_jepa:
            self.target.eval()  # target never uses dropout
        return self

    @staticmethod
    def _states(enc, input_ids, attention_mask):
        return enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def predict_va(self, input_ids, attention_mask) -> torch.Tensor:
        """VA in normalized units: (score - 5) / 4, i.e. roughly [-1, 1]."""
        h = self._states(self.encoder, input_ids, attention_mask)
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled.float())

    def jepa_loss(self, input_ids, attention_mask, jepa_mask) -> tuple[torch.Tensor, dict]:
        jepa_mask = jepa_mask & attention_mask.bool()
        if not jepa_mask.any():
            zero = input_ids.new_zeros((), dtype=torch.float)
            return zero, {"target_std": float("nan")}
        with torch.no_grad():
            t = self._states(self.target, input_ids, attention_mask).float()
            t = F.layer_norm(t, t.shape[-1:])
        ctx_attention = attention_mask * (~jepa_mask).to(attention_mask.dtype)
        ctx = self._states(self.encoder, input_ids, ctx_attention)
        pred = self.predictor(ctx.float(), attention_mask, jepa_mask)
        tgt = t[jepa_mask]
        loss = F.smooth_l1_loss(pred[jepa_mask], tgt)
        # collapse monitor: per-feature std of targets across masked tokens (≈1 is healthy)
        std = tgt.std(dim=0).mean().item() if tgt.size(0) > 1 else float("nan")
        return loss, {"target_std": std}

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        if not self.use_jepa or momentum >= 1.0:
            return
        for pt, pc in zip(self.target.parameters(), self.encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1.0 - momentum)
