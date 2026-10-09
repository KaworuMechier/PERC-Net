"""iTransformer (Liu et al., ICLR 2024) — inverted transformer, ported."""

import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, inverse=False):
        if not inverse:
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (x.var(dim=1, keepdim=True, unbiased=False)
                           + self.eps).sqrt().detach()
            return (x - self.mean) / self.stdev
        return x * self.stdev + self.mean


class InvertedEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.channel_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x, _ = self.channel_attn(x, x, x)
        x = self.norm1(x + residual)
        residual = x
        x = self.time_ffn(x)
        return self.norm2(x + residual)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        d_model = getattr(config, 'd_model', 128)
        n_heads = getattr(config, 'n_heads', 8)
        n_layers = getattr(config, 'n_layers', 2)
        d_ff = getattr(config, 'd_ff', 256)
        dropout = getattr(config, 'dropout', 0.1)

        self.revin = RevIN(self.enc_in)
        self.input_embed = nn.Linear(self.seq_len, d_model)
        self.encoder_layers = nn.ModuleList([
            InvertedEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)])
        self.predictor = nn.Linear(d_model, self.pred_len)

    def forward(self, x):
        x = self.revin(x)
        z = self.input_embed(x.permute(0, 2, 1))      # (B, C, d)
        for layer in self.encoder_layers:
            z = layer(z)
        y = self.predictor(z).permute(0, 2, 1)         # (B, H, C)
        return self.revin(y, inverse=True)
