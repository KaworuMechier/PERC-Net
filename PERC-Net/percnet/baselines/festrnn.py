"""FESTRNN (Wang et al., IEEE TII 2026) — faithful reimplementation.

Reference: "Frequency-Enhanced Dual-Stream Parallel Network for Multienergy
Load Forecasting", DOI 10.1109/TII.2026.3660354.
Code: github.com/balloonplacewzy/FESTRNN.  The released model file has
dimension bugs (hard-coded endlinear(24,12) / finallinear(64,32,3), and
seg_num_x = pred_len//seg_len instead of seq_len//seg_len) that prevent
running at the paper's 24-96h horizons; fixed here per the paper's eqs 3-24.

Architecture:
  RevIN(affine=False) → EESM: spectrum flip × learnable mask, added back
  → Stream 1 (STlinear): moving-avg decomp → seasonal/trend MLPs
  → Stream 2 (PRNN, SegRNN-style): last-value subtract → segment embed →
    GRU encode → positional/channel embedding → GRU decode → add back
  → Energy Reduction: subtract freq_linear(flipped-masked input spectrum)
    from each stream's output spectrum (complex linear K→K_pred)
  → Concat + MLP fusion → all channels (runner slices the 3 targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (B, L, C); endpoint replication padding
        k = self.kernel_size
        front = x[:, :1, :].repeat(1, (k - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (k - 1) // 2, 1)
        xp = torch.cat([front, x, end], dim=1)
        trend = F.avg_pool1d(xp.transpose(1, 2), kernel_size=k,
                             stride=1).transpose(1, 2)
        return x - trend, trend


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        hidden = getattr(config, 'hidden_size', 24)       # STlinear width
        d_model = getattr(config, 'd_model', 64)
        self.d_model = d_model
        self.seg_len = getattr(config, 'seg_len', 12)
        dropout = getattr(config, 'dropout', 0.1)
        kernel = getattr(config, 'decomp_kernel', 13)

        assert self.seq_len % self.seg_len == 0, 'seq_len % seg_len != 0'
        assert self.pred_len % self.seg_len == 0, 'pred_len % seg_len != 0'
        self.seg_num_x = self.seq_len // self.seg_len     # fixed vs release
        self.seg_num_y = self.pred_len // self.seg_len

        self.revin = RevIN(self.enc_in)
        K = self.seq_len // 2 + 1
        self.K_pred = self.pred_len // 2 + 1

        # EESM: learnable per-channel mask on the flipped spectrum (ones init)
        self.mask_matrix = nn.Parameter(torch.ones(K, self.enc_in))
        # Energy Reduction: complex linear map (input spectrum → output spec)
        self.freq_linear = nn.Parameter(
            torch.randn(K, self.K_pred, dtype=torch.cfloat) / K ** 0.5)

        # Stream 1: STlinear
        self.decomp = MovingAvg(kernel)
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, self.pred_len))
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, self.pred_len))

        # Stream 2: PRNN (SegRNN-style patch GRU)
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, d_model), nn.ReLU())
        self.rnn = nn.GRU(d_model, d_model, num_layers=1, bias=True,
                          batch_first=True)
        self.pos_emb = nn.Parameter(
            torch.randn(self.seg_num_y, d_model // 2) * 0.02)
        self.channel_emb = nn.Parameter(
            torch.randn(self.enc_in, d_model // 2) * 0.02)
        self.predict = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, self.seg_len))

        # Fusion (eq. 24): concat two streams along channel → MLP → channels
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.enc_in, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, self.enc_in))

    def _reduce(self, y, x_inverse_fft):
        """Energy reduction (paper eqs. 8-11): subtract the projected
        enhanced-input spectrum from the stream output spectrum.
        y: (B, T, C); x_inverse_fft: (B, K, C) → (B, T, C)."""
        out_fft = torch.fft.rfft(y, dim=1)                          # (B, Kp, C)
        excess = torch.einsum('bkc,kp->bpc',
                              x_inverse_fft, self.freq_linear)
        return torch.fft.irfft(out_fft - excess, n=self.pred_len, dim=1)

    def forward(self, x):
        # x: (B, L, C) — time dim = 1 throughout (as in the paper)
        xn = self.revin(x)

        # EESM (eqs. 3-7)
        x_fft = torch.fft.rfft(xn, dim=1)                           # (B, K, C)
        x_inverse_fft = torch.flip(x_fft, dims=[1]) * self.mask_matrix
        x_ea = torch.fft.irfft(x_fft + x_inverse_fft,
                               n=self.seq_len, dim=1)               # (B, L, C)

        # Stream 1: STlinear (eqs. 12-15) — linear maps over the time axis
        seasonal, trend = self.decomp(x_ea)
        z = (self.linear_seasonal(seasonal.transpose(1, 2)) +
             self.linear_trend(trend.transpose(1, 2))).transpose(1, 2)
        z = self._reduce(z, x_inverse_fft)                          # (B, T, C)

        # Stream 2: PRNN (eqs. 16-23)
        seq_last = x_ea[:, -1:, :].detach()                         # (B, 1, C)
        xr = (x_ea - seq_last).permute(0, 2, 1)                     # (B, C, L)
        emb = self.valueEmbedding(
            xr.reshape(-1, self.seg_num_x, self.seg_len))           # (BC,Nx,d)
        _, hn = self.rnn(emb)
        pos = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.pos_emb.shape[1] * 2)
        _, hy = self.rnn(pos.repeat(x.shape[0], 1, 1),
                         hn.repeat(1, 1, self.seg_num_y).view(
                             1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
        y = (y + seq_last.permute(0, 2, 1)).transpose(1, 2)         # (B, T, C)
        y = self._reduce(y, x_inverse_fft)

        # Fusion (eq. 24): concat along channel → MLP → channels
        out = self.fusion(torch.cat([z, y], dim=-1))                # (B, T, C)
        return self.revin(out, inverse=True)
