"""Crossformer (Zhang & Yan, ICLR 2023) — compact two-stage attention.

Encoder alternates cross-time (TSA) and cross-dimension (CSA) attention over
segment tokens; a compact decoder maps each (segment, dimension) token to its
forecast slice.  Single-scale (segment merging omitted) for the budget table.
"""

import torch
import torch.nn as nn


class TwoStageAttention(nn.Module):
    def __init__(self, n_dims, n_segs, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.tsa = nn.MultiheadAttention(d_model, n_heads,
                                         dropout=dropout, batch_first=True)
        self.csa = nn.MultiheadAttention(d_model, n_heads,
                                         dropout=dropout, batch_first=True)
        self.tsa_norm = nn.LayerNorm(d_model)
        self.csa_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_model * 2, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.n_dims, self.n_segs = n_dims, n_segs

    def forward(self, x):
        # x: (B, S, D, d)
        B, S, D, d = x.shape
        # TSA: attention over segments, per dimension
        xt = x.permute(0, 2, 1, 3).reshape(B * D, S, d)
        xt = self.tsa_norm(xt + self.tsa(xt, xt, xt)[0])
        # CSA: attention over dimensions, per segment
        xd = xt.reshape(B, D, S, d).permute(0, 2, 1, 3).reshape(B * S, D, d)
        xd = self.csa_norm(xd + self.csa(xd, xd, xd)[0])
        # FFN
        xd = self.ffn_norm(xd + self.ffn(xd))
        return xd.reshape(B, S, D, d).permute(0, 2, 1, 3)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        seg_count = getattr(config, 'seg_count', 6)
        d_model = getattr(config, 'd_model', 64)
        n_heads = getattr(config, 'n_heads', 4)
        n_layers = getattr(config, 'n_layers', 2)
        dropout = getattr(config, 'dropout', 0.1)
        self.n_segs = seg_count
        self.seg_len = self.seq_len // seg_count
        self.h_per_seg = self.pred_len // seg_count
        assert self.seq_len % seg_count == 0, 'seq_len % seg_count != 0'
        assert self.pred_len % seg_count == 0, 'pred_len % seg_count != 0'

        self.embed = nn.Linear(self.seg_len, d_model)
        self.seg_pos = nn.Parameter(torch.randn(self.n_segs, d_model) * 0.02)
        self.dim_emb = nn.Parameter(torch.randn(self.enc_in, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TwoStageAttention(self.enc_in, self.n_segs, d_model, n_heads,
                              dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.h_per_seg)

    def forward(self, x):
        B, L, C = x.shape
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False)
               + 1e-5).sqrt().detach()
        xn = (x - mean) / std
        segs = xn.reshape(B, self.n_segs, self.seg_len, C)
        z = self.embed(segs.permute(0, 1, 3, 2))               # (B, S, C, d)
        z = z + self.seg_pos[None, :, None, :] + self.dim_emb[None, None, :, :]
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        out = self.head(z)                                      # (B, S, C, h/S)
        y = out.permute(0, 2, 1, 3).reshape(B, C, self.pred_len)
        return (y.permute(0, 2, 1)) * std + mean
