"""S-Mamba style baseline (Wang et al., 2024) — S6 selective scan.

Minimal, dependency-free implementation: the selective scan uses a
time-varying diagonal (A_t, B_t) recurrence evaluated with a Hillis-Steele
prefix scan (equivalent to S6 without the hardware-aware kernel).  Variate
tokens are mixed by bidirectional Mamba layers over the channel axis
(S-Mamba design).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def tv_scan(a, u):
    """h[t] = a[t]·h[t-1] + u[t], time-varying a — Hillis-Steele, real.
    a, u: (B, L, D, N)"""
    L = a.shape[1]
    idx = torch.arange(L, device=a.device)
    for level in range((L - 1).bit_length()):
        gap = 1 << level
        mask = (idx >= gap).view(1, L, 1, 1)
        a_prev = torch.roll(a, shifts=gap, dims=1)
        u_prev = torch.roll(u, shifts=gap, dims=1)
        a_old = a
        a = torch.where(mask, a * a_prev, a)
        u = torch.where(mask, u + a_old * u_prev, u)
    return u


class MambaLayer(nn.Module):
    """One S6 block over the token axis (input (B, T, D) → (B, T, D))."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, x):
        B, T, D = x.shape
        N = self.d_state
        delta = F.softplus(self.dt_proj(x))                     # (B, T, D)
        A = -torch.exp(self.A_log)                              # (D, N)
        a = torch.exp(delta.unsqueeze(-1) * A)                  # (B, T, D, N)
        BC = self.x_proj(x)
        Bm, Cm = BC.split(N, dim=-1)                            # (B, T, N)
        u = x.unsqueeze(-1) * Bm.unsqueeze(2)                   # (B, T, D, N)
        h = tv_scan(a, u)
        y = (h * Cm.unsqueeze(2)).sum(-1)                       # (B, T, D)
        return y + self.D * x

    def forward(self, x):
        residual = x
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = self.conv1d(x_in.transpose(1, 2))[..., :x.shape[1]]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)
        y = self._scan(x_in)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return self.norm(self.dropout(out) + residual)


class Model(nn.Module):
    """S-Mamba: variate tokens → bidirectional Mamba layers → head."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        d_model = getattr(config, 'd_model', 64)
        n_layers = getattr(config, 'n_layers', 2)
        d_state = getattr(config, 'd_state', 16)
        dropout = getattr(config, 'dropout', 0.1)

        self.embed = nn.Linear(self.seq_len, d_model)
        self.variate_emb = nn.Parameter(torch.randn(self.enc_in, d_model) * 0.02)
        self.layers_fwd = nn.ModuleList([
            MambaLayer(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)])
        self.layers_bwd = nn.ModuleList([
            MambaLayer(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.pred_len)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False)
               + 1e-5).sqrt().detach()
        xn = (x - mean) / std
        z = self.embed(xn.permute(0, 2, 1)) + self.variate_emb  # (B, C, d)
        for lf, lb in zip(self.layers_fwd, self.layers_bwd):
            z = lf(z) + torch.flip(lb(torch.flip(z, dims=[1])), dims=[1])
        z = self.norm(z)
        y = self.head(z).permute(0, 2, 1)                      # (B, H, C)
        return y * std + mean
