"""FiLM (Zhou et al., NeurIPS 2023) — compact Legendre-Fourier mixer.

RevIN → LPU (learnable periodic conv) → Legendre projection (fixed basis) →
FourierMix blocks (frequency-domain complex MLP) → Legendre reconstruction →
linear head.
"""

import math
import torch
import torch.nn as nn


def legendre_basis(L, degree):
    """(L, degree+1) fixed Legendre polynomial matrix on [-1, 1]."""
    t = torch.linspace(-1.0, 1.0, L).unsqueeze(-1)              # (L, 1)
    cols = [torch.ones_like(t), t]
    for n in range(2, degree + 1):
        cols.append(((2 * n - 1) * t * cols[-1] -
                     (n - 1) * cols[-2]) / n)
    return torch.cat(cols, dim=-1)                              # (L, N)


class LPU(nn.Module):
    """Learnable periodic util: causal depthwise conv with period kernel."""

    def __init__(self, channels, period):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, period, groups=channels,
                              padding=period - 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)


class FourierMix(nn.Module):
    """Frequency-domain complex MLP on the projected sequence."""

    def __init__(self, d_modes, dropout=0.1):
        super().__init__()
        self.w_r = nn.Parameter(torch.randn(d_modes, d_modes) /
                                math.sqrt(d_modes))
        self.w_i = nn.Parameter(torch.randn(d_modes, d_modes) /
                                math.sqrt(d_modes))
        self.b_r = nn.Parameter(torch.zeros(d_modes))
        self.b_i = nn.Parameter(torch.zeros(d_modes))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, N_coef) — Legendre coefficients, mixed linearly
        return self.drop(x @ self.w_r + self.b_r)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        degree = getattr(config, 'degree', 24)
        dropout = getattr(config, 'dropout', 0.1)
        period = getattr(config, 'period', 24)

        self.revin_eps = 1e-5
        self.lpu = LPU(self.enc_in, period)
        self.register_buffer('P', legendre_basis(self.seq_len, degree))
        # rec must be a BUFFER (not a plain attribute view of P) — plain
        # tensor attributes are not moved by .to(device) and stay on CPU
        self.register_buffer('rec', self.P.t())
        self.blocks = nn.ModuleList([
            FourierMix(degree + 1, dropout) for _ in range(2)])
        self.norm = nn.LayerNorm(degree + 1)
        self.head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False)
               + self.revin_eps).sqrt().detach()
        x = (x - mean) / std

        x = self.lpu(x)
        coef = torch.einsum('blc,lp->bpc', x, self.P)            # (B, N, C)
        z = coef.permute(0, 2, 1)                                # (B, C, N)
        for blk in self.blocks:
            z = blk(z) + z
        z = self.norm(z)
        coef = z.permute(0, 2, 1)                                # (B, N, C)
        x_rec = torch.einsum('bpc,pl->blc', coef, self.rec)
        y = self.head(x_rec.permute(0, 2, 1)).permute(0, 2, 1)   # (B, H, C)
        return y * std + mean
