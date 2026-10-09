"""DLinear (Zeng et al., AAAI 2023) — channel-independent linear decomposition."""

import torch
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


class MovingAvg(nn.Module):
    """Moving average with edge-REPLICATION padding (official DLinear) —
    zero padding would contaminate the trend at the window edges, where
    recent steps carry the most predictive weight."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=0)

    def forward(self, x):
        # x: (B, C, T)
        k = self.kernel_size
        front = x[:, :, :1].repeat(1, 1, (k - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (k - 1) // 2)
        xp = torch.cat([front, x, end], dim=2)
        return self.avg(xp)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.individual = getattr(config, 'individual', True)
        kernel = getattr(config, 'moving_avg', 25)

        self.revin = RevIN(self.enc_in)
        self.decomp = SeriesDecomp(kernel)

        mk = (lambda: nn.ModuleList([nn.Linear(self.seq_len, self.pred_len)
                                     for _ in range(self.enc_in)])) \
            if self.individual else \
            (lambda: nn.Linear(self.seq_len, self.pred_len))
        self.Linear_Seasonal = mk()
        self.Linear_Trend = mk()

    def _apply_linear(self, lin, z):
        if self.individual:
            return torch.stack([lin[i](z[:, i, :])
                                for i in range(self.enc_in)], dim=1)
        return lin(z)

    def forward(self, x):
        x = self.revin(x)
        x = x.permute(0, 2, 1)
        seasonal, trend = self.decomp(x)
        y = self._apply_linear(self.Linear_Seasonal, seasonal) + \
            self._apply_linear(self.Linear_Trend, trend)
        return self.revin(y.permute(0, 2, 1), inverse=True)
