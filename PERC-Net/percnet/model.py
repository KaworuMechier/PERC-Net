"""PERC-Net — final configuration, self-contained reference implementation
===========================================================================

This file is the model that produces every number reported in the paper.
It is a *frozen* single-configuration extract of the development file: the
ablation switches, the multi-horizon wrapper and the unused branches have
been removed, so that what is read here is exactly what was run.

The switchable variant behind the ablation table lives in
``percnet/model_ablation.py``.  Under the paper configuration that variant
is identical to this file — same parameter count, same forward output —
which ``scripts/smoke_test.py`` verifies.

Configuration (the "joint" setting of the paper)
------------------------------------------------
    component specialist chain + exclusive-target training        (Sec. 2.3-2.4)
    joint probabilistic training: prob_detach = False            (Sec. 2.5)
    lead-time-conditioned probabilistic head                     (Sec. 2.6)
    exact CEL identity output layer, zero parameters             (Sec. 2.7)
    nwp setting: the T future rows are fed to the weather branch (Sec. 3.5)

Input layout (20 channels; see Table A of the paper)
----------------------------------------------------
    indices  0- 2   loads            KW, CHWTON, HTMBTU      <- only these are window-normalised
    indices  3- 9   weather          TEMP DPT SLP SEA-P WBT ALT DOY-COS
    indices 10-11   derived          CMBMBBTU, GHG            <- masked on future rows
    indices 12-15   calendar         sin/cos(HOD), sin/cos(DOW)
    indices 16-19   rolling weather  TEMP/WBT 24 h and 72 h means

Shapes: x is (B, seq_len + pred_len, 20); the first ``seq_len`` rows are
history, the last ``pred_len`` rows are the future covariates.

Quick check
-----------
    python percnet/model.py

builds each of the four horizons, runs a forward and a backward pass, and
prints the parameter count, the forward cost and the structural invariants.

Correspondence with the paper
-----------------------------
    Sec. 2.2  statistical prior externalisation  -> _prepare(): RevIN, cycle
              queues, last-value anchor
    Sec. 2.3  component specialist chain         -> forward(): seasonal/trend
              heads, patch-GRU local stream, weather branch, additive rail
    Sec. 2.4  exclusive-target training          -> compute_loss(): the four
              detached component residuals
    Sec. 2.5  probabilistic head, delivery       -> quantile/scale heads; the
              delivered point forecast is the tau = 0.5 quantile
    Sec. 2.6  lead-time-conditioned head         -> horizon_emb
    Sec. 2.7  CEL identity output layer          -> the ``cmbbtu`` output
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Frozen hyper-parameters (24 h configuration of the paper)
# ─────────────────────────────────────────────────────────────────────────────
FINAL = dict(
    seq_len=168,            # one week of history
    pred_len=24,            # 24 / 48 / 72 / 96 in the paper
    enc_in=20,              # input channels, see the layout above
    n_free=3,               # free (load) target channels
    hidden=36,              # backbone width (Table B)
    decomp_kernel=13,       # moving-average kernel for trend/seasonal split
    seg_len=24,             # patch length of the local stream
    d_model=12,             # patch-GRU hidden width
    dropout=0.1,
    n_quantiles=9,          # tau = 0.05 ... 0.95, uniform grid
    beta=0.9,               # CVaR tail level
    lam_pin=1.0,            # pinball weight
    lam_nll=1.0,            # Laplace-NLL weight on the scale head
    lam_tail=0.5,           # CVaR weight
    lam_mono=0.1,           # quantile monotonicity penalty
    lam_dec=10.0,           # component (exclusive-target) loss weight
)

CAL_SLICE = slice(12, 16)   # sin/cos(HOD), sin/cos(DOW) inside the input
CEL_COEF = (0.003412, 0.0120, 1.0000)  # ASU metering unit conversions


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────────────────────
class RevINLoads(nn.Module):
    """Window-adaptive reversible instance normalisation (Sec. 2.2).

    Statistics are computed on the load channels only; the exogenous channels
    keep their static scale, because the absolute level of a weather variable
    is physically informative (38 C and 22 C imply very different cooling
    loads) and full-channel normalisation would erase it.

    ``forward`` stores the window mean/std as attributes so that the inverse
    transform can be applied later in the forward pass; ``compute_loss`` also
    reads them to rebuild the target in the same normalised frame.
    """

    def __init__(self, n_loads=3, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.n_loads = n_loads
        self.mean = None
        self.stdev = None

    def forward(self, x, stat_len=None):
        nl = self.n_loads
        xl, xe = x[:, :, :nl], x[:, :, nl:]
        xs = xl if stat_len is None else xl[:, :stat_len]
        self.mean = xs.mean(dim=1, keepdim=True).detach()
        self.stdev = (xs.var(dim=1, keepdim=True, unbiased=False)
                      + self.eps).sqrt().detach()
        return torch.cat([(xl - self.mean) / self.stdev, xe], dim=-1)

    def inverse(self, x):
        return x * self.stdev + self.mean


class _SeriesDecomp(nn.Module):
    """Moving-average decomposition (Sec. 2.3): returns (residual, trend)."""

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


class PatchGRUStream(nn.Module):
    """Local stream: one-shot parallel-decoding patch GRU (Sec. 2.3).

    Each channel is split into patches of ``seg_len`` steps, encoded by a
    shared linear layer and compressed by a GRU into a single hidden state.
    A decoder driven by positional and channel embeddings then emits every
    prediction patch in a single forward pass, so there is no recursive
    rollout and no error accumulation. It also exposes a pooled hidden
    feature for the probabilistic head.
    """

    def __init__(self, seq_len, pred_len, n_chan, seg_len=24, d_model=12,
                 dropout=0.1):
        super().__init__()
        assert seq_len % seg_len == 0 and pred_len % seg_len == 0
        self.seg_len = seg_len
        self.seg_num_x = seq_len // seg_len
        self.seg_num_y = pred_len // seg_len
        self.n_chan = n_chan
        self.d_model = d_model

        self.value_emb = nn.Sequential(nn.Linear(seg_len, d_model), nn.ReLU())
        self.rnn = nn.GRU(input_size=d_model, hidden_size=d_model,
                          num_layers=1, batch_first=True)
        self.pos_emb = nn.Parameter(
            torch.randn(self.seg_num_y, d_model // 2) * 0.02)
        self.chan_emb = nn.Parameter(
            torch.randn(n_chan, d_model // 2) * 0.02)
        self.predict = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(d_model, seg_len))

    def forward(self, x):
        B, S, C = x.shape
        xn = x.permute(0, 2, 1).reshape(B * C, self.seg_num_x, self.seg_len)
        _, hn = self.rnn(self.value_emb(xn))              # (1, B*C, d)
        pos = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(C, 1, 1),
            self.chan_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
        ], dim=-1).view(-1, 1, self.d_model)
        _, hy = self.rnn(pos.repeat(B, 1, 1),
                         hn.repeat(1, 1, self.seg_num_y)
                         .view(1, -1, self.d_model))
        y = self.predict(hy).view(B, C, self.seg_num_y * self.seg_len)
        feat = hy.view(B, C, self.seg_num_y, self.d_model).mean(dim=(1, 2))
        return y.permute(0, 2, 1), feat                 # (B, H, C), (B, d)


# ─────────────────────────────────────────────────────────────────────────────
# The model
# ─────────────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    """PERC-Net, final (joint) configuration.

    ``forward(x)`` returns a dict with

        mu        (B, T, D)     point forecast = predictive median (tau = 0.5)
        quantiles (B, T, D, K)  K = 9 quantiles on the [0.05, 0.95] grid
        sigma     (B, T, D)     per-channel, per-lead-time scale
        cmbbtu    (B, T, 1)     system-level energy, exactly consistent with mu

    and, for the loss only,

        _streams  the four component outputs, and
        _targets  the pieces needed to rebuild the target in the same
                  de-cycled / de-anchored frame.
    """

    def __init__(self, seq_len=168, pred_len=24, enc_in=20, n_free=3,
                 hidden=36, decomp_kernel=13, seg_len=24, d_model=12,
                 dropout=0.1, n_quantiles=9, beta=0.9, lam_pin=1.0,
                 lam_nll=1.0, lam_tail=0.5, lam_mono=0.1, lam_dec=10.0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.n_free = n_free
        self.K = n_quantiles
        self.taus = torch.linspace(0.05, 0.95, n_quantiles)
        self.beta = beta
        self.lam_pin, self.lam_nll = lam_pin, lam_nll
        self.lam_tail, self.lam_mono, self.lam_dec = lam_tail, lam_mono, lam_dec

        self.revin = RevINLoads(n_free)
        self.decomp = _SeriesDecomp(decomp_kernel)

        # specialists 1 and 2: decomposition-matched linear heads (Sec. 2.3)
        self.linear_seasonal = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, pred_len))
        self.linear_trend = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, pred_len))

        # specialist 3: local stream on the linear residual
        self.prnn = PatchGRUStream(seq_len, pred_len, n_free, seg_len,
                                   d_model, dropout)

        # specialist 4: weather branch on the future rows. The last layer is
        # zero-initialised, so it starts as the exact zero function and earns
        # its output against the residual left by the preceding specialists.
        n_exog = enc_in - n_free
        self.future_branch = nn.Sequential(
            nn.Linear(n_exog, 32), nn.GELU(), nn.Linear(32, n_free))
        nn.init.zeros_(self.future_branch[-1].weight)
        nn.init.zeros_(self.future_branch[-1].bias)

        # statistical priors, placed outside the backbone (Sec. 2.2)
        self.cycle_daily = nn.Parameter(torch.zeros(24, n_free))
        self.cycle_weekly = nn.Parameter(torch.zeros(168, n_free))

        # probabilistic head (Sec. 2.5-2.6). The pooled feature carries no
        # lead-time information, so a zero-initialised per-step embedding
        # restores the lead-time axis; the head weights themselves stay
        # horizon-invariant, i.e. O(1) in the forecast horizon.
        self.ctx_proj = nn.Linear(hidden + d_model, hidden)
        self.horizon_emb = nn.Parameter(torch.zeros(pred_len, hidden))
        self.quantile_head = nn.Linear(hidden, n_free * n_quantiles)
        self.scale_head = nn.Sequential(nn.Linear(hidden, n_free),
                                        nn.Softplus())
        nn.init.constant_(self.scale_head[0].bias, 5.0)
        self.cvar_t = nn.Parameter(torch.tensor(1.0))

        # CEL identity coefficients: unit conversions for the ASU metering
        # units, not fitted parameters (Sec. 2.7)
        self.register_buffer('cel_coef', torch.tensor(CEL_COEF))
        self.register_buffer('load_scale', torch.ones(n_free))

    # ── Sec. 2.2: RevIN, cycle queues, last-value anchor ────────────────────
    def _prepare(self, x):
        xn = self.revin(x, stat_len=self.seq_len)
        nf = self.n_free

        # Cycle queues are indexed by the phase recovered from the calendar
        # channels, so the model does not need to know the sequence start.
        with torch.no_grad():
            cal = xn[:, self.seq_len - 1, CAL_SLICE]
            hod = torch.remainder(torch.round(
                torch.atan2(cal[:, 0], cal[:, 1]) / (2 * math.pi) * 24),
                24).long()
            dow = torch.remainder(torch.round(
                torch.atan2(cal[:, 2], cal[:, 3]) / (2 * math.pi) * 7),
                7).long()
            how_last = hod + 24 * dow
        steps = torch.arange(1, self.pred_len + 1, device=x.device)
        hod_f = torch.remainder(hod.view(-1, 1) + steps.view(1, -1), 24)
        hofw_f = torch.remainder(how_last.view(-1, 1) + steps.view(1, -1), 168)
        cyc_d = self.cycle_daily[hod_f]                  # (B, T, nf)
        cyc_w = self.cycle_weekly[hofw_f]                # (B, T, nf)

        past = xn[:, :self.seq_len]
        cal_p = past[:, :, CAL_SLICE]
        with torch.no_grad():
            hod_p = torch.remainder(torch.round(
                torch.atan2(cal_p[..., 0], cal_p[..., 1])
                / (2 * math.pi) * 24), 24).long()
            dow_p = torch.remainder(torch.round(
                torch.atan2(cal_p[..., 2], cal_p[..., 3])
                / (2 * math.pi) * 7), 7).long()
            hofw_p = torch.remainder(hod_p + 24 * dow_p, 168)
        past = torch.cat([past[:, :, :nf]
                          - self.cycle_daily[hod_p]
                          - self.cycle_weekly[hofw_p],
                          past[:, :, nf:]], dim=-1)

        # last-value anchor: the backbone predicts only the deviation from the
        # final observation of the window (zero parameters)
        seq_last = past[:, -1:, :nf].detach()
        past = torch.cat([past[:, :, :nf] - seq_last, past[:, :, nf:]],
                         dim=-1)
        return xn, past, seq_last, cyc_d, cyc_w

    # ── Sec. 2.3-2.7: forward path ──────────────────────────────────────────
    def forward(self, x):
        xn, past, seq_last, cyc_d, cyc_w = self._prepare(x)
        nf = self.n_free
        residual = past[:, :, :nf]                       # (B, S, nf)

        # specialists 1 and 2: linear heads on the decomposed components
        seasonal, trend = self.decomp(residual)
        s = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        t = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)

        # specialist 3: local stream on the same residual
        loc, feat_loc = self.prnn(residual)

        # additive rail
        mu = s + t + loc

        # specialist 4: weather branch on the future rows
        fut = self.future_branch(xn[:, self.seq_len:, nf:])
        mu = mu + fut

        # pooled head feature; the projection is horizon-invariant
        feat_seas = self.linear_seasonal[0](seasonal.transpose(1, 2))
        feat_seas = F.leaky_relu(feat_seas).mean(dim=1)          # (B, hidden)
        feat = self.ctx_proj(torch.cat([feat_seas, feat_loc], dim=-1))

        # Sec. 2.6: restore the lead-time axis, then run the head once per step
        fh = feat.unsqueeze(1) + self.horizon_emb.unsqueeze(0)   # (B, T, hidden)
        q = self.quantile_head(fh).view(-1, self.pred_len, nf, self.K)
        sigma = self.scale_head(fh) + 1e-3                       # (B, T, nf)

        # joint configuration: the quantile offsets are NOT detached from mu,
        # so pinball / NLL / CVaR share the backbone gradient with MAPE
        q = q + mu.unsqueeze(-1)                                 # (B, T, nf, K)

        # add the priors back, then invert the window normalisation
        mu = mu + cyc_d + cyc_w
        q = q + (cyc_d + cyc_w).unsqueeze(-1)
        mu = mu + seq_last
        q = q + seq_last.unsqueeze(-1)

        win_std = self.revin.stdev[:, :1, :nf]
        win_mean = self.revin.mean[:, :1, :nf]
        mu = mu * win_std + win_mean
        q = q * win_std.unsqueeze(-1) + win_mean.unsqueeze(-1)

        # Sec. 2.7: CEL identity output layer — zero parameters, and the
        # consistency error between mu and cmbbtu is zero by construction
        mu_scaled = mu * self.load_scale.view(1, 1, nf)
        cmbbtu = (mu_scaled * (self.cel_coef * self.load_scale)
                  .view(1, 1, nf)).sum(dim=-1, keepdim=True)

        return {
            'mu': mu, 'quantiles': q, 'sigma': sigma, 'cmbbtu': cmbbtu,
            '_streams': {'seasonal': s, 'trend': t, 'local': loc,
                         'weather': fut},
            # target-frame pieces, consumed by compute_loss
            '_targets': {'cyc_fut': cyc_d + cyc_w, 'seq_last': seq_last,
                         'win_std': win_std, 'win_mean': win_mean},
        }

    # ── Sec. 2.4-2.6: joint objective with exclusive component targets ─────
    def compute_loss(self, out, y_true, scale3):
        """y_true is the column-normalised target; scale3 maps it back to
        physical units per channel."""
        nf = self.n_free
        mu, q, sigma = out['mu'], out['quantiles'], out['sigma']
        streams = out['_streams']
        tg = out['_targets']
        y_raw = y_true * scale3
        q_raw = q * scale3.view(1, 1, nf, 1)
        mu_raw = mu * scale3

        # --- shared paths -------------------------------------------------
        l_point = ((y_raw - mu_raw).abs()
                   / (y_raw.abs() + 1e-2)).mean() * 100          # MAPE

        u = y_raw.unsqueeze(-1) - q_raw
        taus = self.taus.to(u.device).view(1, 1, 1, -1)
        l_pin = torch.maximum(taus * u, (taus - 1) * u).mean() * self.lam_pin

        z = (y_raw - mu_raw).abs() / sigma.clamp(min=1e-3)       # joint: no detach
        l_nll = (z + torch.log(sigma.clamp(min=1e-3))).mean()
        l_cvar = self.cvar_t + F.relu(z - self.cvar_t).mean() / (1 - self.beta)
        l_mono = F.relu(-torch.diff(q, dim=-1)).mean()

        # --- exclusive component targets (Sec. 2.4) -----------------------
        # rebuild the target in the de-cycled, de-anchored frame. NOTE units:
        # y_true is column-normalised and win_mean/win_std are window
        # statistics of that same normalised space, so y_true is used here.
        r_full = (y_true - tg['win_mean']) / tg['win_std'] - tg['cyc_fut'] \
            - tg['seq_last']
        r_full = r_full.detach()                    # target: no gradient

        s, t, loc = streams['seasonal'], streams['trend'], streams['local']
        seas_tgt, trend_tgt = self.decomp(r_full)
        l_seas = (s - seas_tgt).abs().mean()
        l_trend = (t - trend_tgt).abs().mean()
        lin_out = (s + t).detach()                  # gradient-boosting-style
        l_loc = (loc - (r_full - lin_out)).abs().mean()
        l_weather = (streams['weather']
                     - (r_full - (lin_out + loc.detach()))).abs().mean()
        l_comp = l_seas + l_trend + l_loc + l_weather

        return (l_pin + l_point
                + self.lam_nll * l_nll
                + self.lam_tail * l_cvar
                + self.lam_mono * l_mono
                + self.lam_dec * l_comp)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / structural invariants
# ─────────────────────────────────────────────────────────────────────────────
def _macs(model, pred_len, batch=1):
    """Forward multiply-accumulates per sample, counted along the real
    execution path (the lead-time-conditioned head runs once per lead time)."""
    from torch.utils.flop_counter import FlopCounterMode
    x1 = torch.randn(batch, FINAL['seq_len'] + pred_len, FINAL['enc_in'])
    with torch.no_grad():
        model(x1)
        c = FlopCounterMode(display=False)
        with c:
            model(x1)
    return c.get_total_flops() / 2 / 1e6 / batch


def _report(pred_len):
    torch.manual_seed(0)
    m = Model(pred_len=pred_len).eval()
    n_param = sum(p.numel() for p in m.parameters())
    x = torch.randn(4, FINAL['seq_len'] + pred_len, FINAL['enc_in'])
    with torch.no_grad():
        out = m(x)
    mu, q, sig, cm = out['mu'], out['quantiles'], out['sigma'], out['cmbbtu']

    # (1) CEL identity: exact by construction, zero parameters (Sec. 2.7)
    cm_ref = (mu * m.load_scale.view(1, 1, -1)
              * (m.cel_coef * m.load_scale).view(1, 1, -1)).sum(-1, keepdim=True)
    cel_err = float((cm - cm_ref).abs().max())

    # (2) Lead-time pathway (Sec. 2.6). horizon_emb is zero-initialised, so at
    #     init the head is *identical* to a pooled head and the fan ratio is
    #     exactly 1.000 — that is the intended initial condition, not a
    #     failure. Perturbing the embedding must move the ratio off 1.000,
    #     which is what shows the pathway is live and trainable.
    def fan_ratio():
        with torch.no_grad():
            qq = m(x)['quantiles']
        w = (qq[..., -1] - qq[..., 0]).mean(dim=(0, 2))
        return float(w[-1] / w[0])

    fan_init = fan_ratio()
    keep = m.horizon_emb.detach().clone()
    with torch.no_grad():
        m.horizon_emb.add_(torch.randn_like(m.horizon_emb) * 0.05)
    fan_pert = fan_ratio()
    with torch.no_grad():
        m.horizon_emb.copy_(keep)

    # (3) objective wiring: every parameter tensor must receive a gradient
    m.train()
    y = torch.randn(4, pred_len, FINAL['n_free'])
    loss = m.compute_loss(m(x), y, torch.ones(FINAL['n_free']))
    loss.backward()
    params = list(m.parameters())
    grads = sum(p.grad is not None for p in params)

    print(f"pred_len = {pred_len:>3} | params {n_param:,} | "
          f"{_macs(m, pred_len):.4f} MMac/sample")
    print(f"   shapes           mu{tuple(mu.shape)}  q{tuple(q.shape)}  "
          f"sigma{tuple(sig.shape)}  cmbbtu{tuple(cm.shape)}")
    print(f"   CEL identity     max|cmbbtu - identity(mu)| = {cel_err:.1e}   "
          f"(exact, 0 parameters)")
    print(f"   lead-time head   fan ratio at init {fan_init:.3f} "
          f"(zero-init => identical to a pooled head); "
          f"after perturbing horizon_emb {fan_pert:.3f}")
    print(f"   objective        loss = {loss.item():.2f}; "
          f"{grads}/{len(params)} parameter tensors received a gradient")


if __name__ == '__main__':
    for h in (24, 48, 72, 96):
        _report(h)
        print()
