"""Shared building blocks used by the ablative variant of PERC-Net.

``percnet.model`` (the paper model) is deliberately self-contained: it
repeats these two classes inline so that the single file a reviewer reads
is exactly the file that was run.  They are factored out here only because
``percnet.model_ablation`` needs a scope-switchable RevIN that the frozen
model does not carry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevINLoads(nn.Module):
    """Window-adaptive reversible instance normalisation.

    ``scope='loads'`` (default) computes the window statistics on the load
    channels only and leaves the exogenous channels at their static scale:
    the absolute level of a weather variable is physically informative
    (38 C and 22 C imply very different cooling loads), and normalising it
    away costs accuracy.

    ``scope='all'`` is the classic full-channel RevIN, kept only so the
    ablation table can quantify what the load-only scope buys.

    ``forward`` stashes the window mean/std as attributes so the inverse
    transform -- and ``compute_loss``, which rebuilds the target in the
    same normalised frame -- can read them back.
    """

    def __init__(self, n_loads=3, scope='loads', eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scope = scope
        self.n_loads = n_loads

    def forward(self, x, stat_len=None, inverse=False):
        if self.scope == 'all':
            xs = x if stat_len is None else x[:, :stat_len]
            self.mean = xs.mean(dim=1, keepdim=True).detach()
            self.stdev = (xs.var(dim=1, keepdim=True, unbiased=False)
                          + self.eps).sqrt().detach()
            if not inverse:
                return (x - self.mean) / self.stdev
            return x * self.stdev + self.mean
        nl = self.n_loads
        xl, xe = x[:, :, :nl], x[:, :, nl:]
        if not inverse:
            xs = xl if stat_len is None else xl[:, :stat_len]
            self.mean = xs.mean(dim=1, keepdim=True).detach()
            self.stdev = (xs.var(dim=1, keepdim=True, unbiased=False)
                          + self.eps).sqrt().detach()
            return torch.cat([(xl - self.mean) / self.stdev, xe], dim=-1)
        xl = xl * self.stdev + self.mean
        return torch.cat([xl, xe], dim=-1)


class _SeriesDecomp(nn.Module):
    """Moving-average decomposition: ``forward(x) -> (residual, trend)``."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        k = self.kernel_size
        front = x[:, :1, :].repeat(1, (k - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (k - 1) // 2, 1)
        xp = torch.cat([front, x, end], dim=1)
        trend = F.avg_pool1d(xp.transpose(1, 2), kernel_size=k,
                             stride=1).transpose(1, 2)
        return x - trend, trend
