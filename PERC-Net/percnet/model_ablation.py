"""PERC-Net v2: Decoupled Component Training (DCT)
===================================================
v2 elevates PERC-Net's core principle — decoupled gradient paths — from
the loss level to the architecture level.  v1 trains one shared point
estimate mu with a single point loss and patches the interference with
zero-init gates (NWP) / detached centers (probabilistic head).  v2
instead composes the forecast as an additive chain of SPECIALISTS, each
with its own target:

    y  =  cycle(t) + anchor + trend(t) + seasonal(t) + local(t) + weather(t)

  trend / seasonal : linear heads supervised against the moving-average
                     decomposition of the de-cycled, anchored target
  local            : patch-GRU stream supervised against the residual
                     left by the linear heads (targets detached —
                     gradient-boosting-style curriculum)
  weather          : NWP branch supervised against the residual left by
                     all previous streams; the zero-init final layer
                     plays the role v1's tanh gate played (no gate param)

The shared MAPE on the full sum keeps the composition aligned with the
deployment metric; the component L1s give each specialist a
well-conditioned target so no multi-objective interference needs to be
patched after the fact.  The probabilistic head is unchanged: quantile
offsets around the detached mu, delivering the predictive median.

Parameter budget (~19K, horizon-invariant): the flatten-projection
ctx_proj (H·C → hidden) is replaced by pooled stream features
(hidden + d → hidden).

forward(x) → dict(mu, quantiles, sigma, cmbbtu) — same contract and
decoupled loss as PERCNet.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from percnet.primitives import RevINLoads, _SeriesDecomp


class PatchGRUStream(nn.Module):
    """SegRNN-style one-shot patch GRU over residual channels.

    Input: (B, S, C) residuals in normalized scale.  Each channel is
    split into seg_len patches, encoded by a shared linear+ReLU and a
    GRU; decoding runs the GRU once over positional+channel embeddings
    to emit all H steps per channel (no recursive rollout).
    Also exposes a pooled hidden feature for the probabilistic head.
    """

    def __init__(self, seq_len, pred_len, n_chan, seg_len=24, d_model=12,
                 dropout=0.1):
        super().__init__()
        assert seq_len % seg_len == 0, 'seq_len must be divisible by seg_len'
        assert pred_len % seg_len == 0, 'pred_len must be divisible by seg_len'
        self.seg_len = seg_len
        self.seg_num_x = seq_len // seg_len
        self.seg_num_y = pred_len // seg_len
        self.n_chan = n_chan
        self.d_model = d_model

        self.value_emb = nn.Sequential(
            nn.Linear(seg_len, d_model), nn.ReLU())
        self.rnn = nn.GRU(input_size=d_model, hidden_size=d_model,
                          num_layers=1, batch_first=True)
        self.pos_emb = nn.Parameter(
            torch.randn(self.seg_num_y, d_model // 2) * 0.02)
        self.chan_emb = nn.Parameter(
            torch.randn(n_chan, d_model // 2) * 0.02)
        self.predict = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, seg_len))

    def forward(self, x):
        # x: (B, S, C)
        B, S, C = x.shape
        xn = x.permute(0, 2, 1).reshape(B * C, self.seg_num_x, self.seg_len)
        _, hn = self.rnn(self.value_emb(xn))            # (1, B*C, d)
        pos = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(C, 1, 1),
            self.chan_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
        ], dim=-1).view(-1, 1, self.d_model)            # (C*seg_y, 1, d)
        _, hy = self.rnn(pos.repeat(B, 1, 1),
                         hn.repeat(1, 1, self.seg_num_y)
                         .view(1, -1, self.d_model))
        y = self.predict(hy).view(B, C, self.seg_num_y * self.seg_len)
        # pooled decoder feature (B, d) — horizon-invariant head feature
        feat = hy.view(B, C, self.seg_num_y, self.d_model).mean(dim=(1, 2))
        return y.permute(0, 2, 1), feat                 # (B, H, C), (B, d)


class Model(nn.Module):
    """PERC-Net v2 — decoupled component training."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.use_future_weather = getattr(config, 'use_future_weather', False)
        self.use_calendar = getattr(config, 'use_calendar', True)
        self.n_free = getattr(config, 'n_free', 3)
        self.physics_identity = getattr(config, 'physics_identity', True)
        K = getattr(config, 'n_quantiles', 9)
        self.K = K
        self.taus = torch.linspace(0.05, 0.95, K)
        self.beta = getattr(config, 'cvar_beta', 0.9)
        self.lam_tail = getattr(config, 'lam_tail', 0.5)
        self.lam_mono = getattr(config, 'lam_mono', 0.1)
        self.lam_nll = getattr(config, 'lam_nll', 1.0)
        # pinball weight.  l_point is MAPE x100 (O(10)) while l_pin lives in
        # normalized space (O(0.1)), so the quantile head is ~40x under-weighted
        # and the intervals come out systematically too narrow (0.88 vs 0.90
        # nominal coverage).  Default 1.0 == previous behaviour.
        self.lam_pin = getattr(config, 'lam_pin', 1.0)
        # component-specialist loss weight (normalized-L1 targets)
        self.lam_dec = getattr(config, 'lam_dec', 10.0)
        hidden = getattr(config, 'linear_hidden', 24)
        kernel = getattr(config, 'decomp_kernel', 13)

        self.revin = RevINLoads(self.n_free,
                                getattr(config, 'revin_scope', 'loads'))
        self.decomp = _SeriesDecomp(kernel)

        # specialists 1+2: seasonal / trend linear heads
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, self.pred_len))
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, self.pred_len))

        # specialist 3: PRNN-lite local stream (fits the linear residual)
        self.prnn = PatchGRUStream(
            self.seq_len, self.pred_len, self.n_free,
            seg_len=getattr(config, 'prnn_seg_len', 24),
            d_model=getattr(config, 'prnn_dim', 12),
            dropout=getattr(config, 'prnn_dropout', 0.1))

        # specialist 4: NWP branch — zero-init final layer replaces v1's
        # gate: it starts as an exact zero function and earns its output
        n_exog = self.enc_in - self.n_free
        self.future_branch = nn.Sequential(
            nn.Linear(n_exog, 32), nn.GELU(),
            nn.Linear(32, self.n_free))
        nn.init.zeros_(self.future_branch[-1].weight)
        nn.init.zeros_(self.future_branch[-1].bias)

        # CycleNet-style learnable cycles (daily + weekly), per target
        self.use_cycle_queue = (getattr(config, 'use_cycle_queue', True)
                                and self.use_calendar)
        if self.use_cycle_queue:
            self.cycle_daily = nn.Parameter(torch.zeros(24, self.n_free))
            self.cycle_weekly = nn.Parameter(torch.zeros(168, self.n_free))
        # Horizon-adaptive cycle gates (v2.1): the ablation showed the
        # cyclic prior helps at 24h (+2.6) but HURTS at 96h (−17.9) —
        # a fixed periodicity becomes bias at long horizons.  Each queue
        # gets a 2-parameter gate g(t) = sigmoid(a − b·t/H) applied to the
        # OUTPUT-side cycle addition: open at the first predicted step,
        # learnably closing toward the horizon end (a=6, b=0 init = fully
        # open, i.e. the v2 behavior is the special case b=0).
        self.cycle_gate = getattr(config, 'cycle_gate', False)
        if self.cycle_gate:
            self.gate_d = nn.Parameter(torch.tensor([6.0, 0.0]))
            self.gate_w = nn.Parameter(torch.tensor([6.0, 0.0]))

        # last-value anchor
        self.use_last_value_anchor = getattr(
            config, 'use_last_value_anchor', True)

        # probabilistic head (unchanged decoupling: offsets around
        # detached mu; delivered point = predictive median tau=0.5)
        self.prob_head = getattr(config, 'prob_head', True)
        self.prob_detach = getattr(config, 'prob_detach', True)
        self.point_loss = getattr(config, 'point_loss', 'mape')
        self.ctx_proj = nn.Linear(hidden + self.prnn.d_model, hidden)
        if self.prob_head:
            self.quantile_head = nn.Linear(hidden, self.n_free * K)
            self.scale_head = nn.Sequential(nn.Linear(hidden, self.n_free),
                                            nn.Softplus())
            nn.init.constant_(self.scale_head[0].bias, 5.0)

        # Horizon-aware probabilistic head.  `feat` is pooled over the
        # sequence (see .mean(dim=1) in forward), so it carries NO lead-time
        # information: a single set of quantile offsets gets broadcast to
        # every horizon and the predictive spread is identical at h=1 and
        # h=96.  A zero-initialised per-step embedding restores the horizon
        # axis while the head itself stays shared, so the parameter count
        # remains O(1) in pred_len.  Zeros init consumes no RNG, so runs are
        # unaffected when the flag is off.
        self.horizon_cond_head = getattr(config, 'horizon_cond_head', False)
        if self.horizon_cond_head:
            self.horizon_emb = nn.Parameter(torch.zeros(self.pred_len, hidden))
        self.cvar_t = nn.Parameter(torch.tensor(1.0))

        # CEL identity coefficients (ASU only)
        self.register_buffer('cel_coef',
                             torch.tensor([0.003412, 0.012, 1.0]))
        self.register_buffer('load_scale', torch.ones(self.n_free))

    # ── shared normalization / de-cycling / anchoring ─────────────────
    def _prepare(self, x):
        stat_len = self.seq_len if self.use_future_weather else None
        xn = self.revin(x, stat_len=stat_len)
        nf = self.n_free
        cyc_d = cyc_w = None
        if self.use_cycle_queue:
            with torch.no_grad():
                cal = xn[:, self.seq_len - 1, 12:16]
                hod = torch.remainder(torch.round(
                    torch.atan2(cal[:, 0], cal[:, 1])
                    / (2 * math.pi) * 24), 24).long()
                dow = torch.remainder(torch.round(
                    torch.atan2(cal[:, 2], cal[:, 3])
                    / (2 * math.pi) * 7), 7).long()
                how_last = hod + 24 * dow
            steps = torch.arange(1, self.pred_len + 1, device=x.device)
            hod_f = torch.remainder(hod.view(-1, 1) + steps.view(1, -1), 24)
            hofw_f = torch.remainder(how_last.view(-1, 1) + steps.view(1, -1),
                                     168)
            cyc_d = self.cycle_daily[hod_f]
            cyc_w = self.cycle_weekly[hofw_f]
            past = xn[:, :self.seq_len]
            cal_p = xn[:, :self.seq_len, 12:16]
            with torch.no_grad():
                hod_p = torch.remainder(torch.round(
                    torch.atan2(cal_p[..., 0], cal_p[..., 1])
                    / (2 * math.pi) * 24), 24).long()
                dow_p = torch.remainder(torch.round(
                    torch.atan2(cal_p[..., 2], cal_p[..., 3])
                    / (2 * math.pi) * 7), 7).long()
                hofw_p = torch.remainder(hod_p + 24 * dow_p, 168)
            past = torch.cat([
                past[:, :, :nf] - self.cycle_daily[hod_p]
                - self.cycle_weekly[hofw_p],
                past[:, :, nf:]], dim=-1)
        else:
            past = xn[:, :self.seq_len]

        seq_last = None
        if self.use_last_value_anchor:
            seq_last = past[:, -1:, :nf].detach()
            past = torch.cat([past[:, :, :nf] - seq_last,
                              past[:, :, nf:]], dim=-1)
        return xn, past, seq_last, cyc_d, cyc_w

    def forward(self, x):
        xn, past, seq_last, cyc_d, cyc_w = self._prepare(x)
        nf = self.n_free
        residual = past[:, :, :nf]                       # (B, S, C)

        # specialists 1+2: decomposition-matched linear heads
        seasonal, trend = self.decomp(residual)
        s = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        t = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)

        # specialist 3: local stream on the linear residual
        loc, feat_loc = self.prnn(residual)

        mu = s + t + loc

        # specialist 4: NWP branch on future weather rows
        fut = None
        if self.use_future_weather:
            fut_exog = xn[:, self.seq_len:, nf:]         # (B, H, C-nf)
            fut = self.future_branch(fut_exog)
            mu = mu + fut

        # probabilistic head: horizon-invariant pooled features
        feat_seas = self.linear_seasonal[0](seasonal.transpose(1, 2))
        feat_seas = F.leaky_relu(feat_seas).mean(dim=1)  # (B, hidden)
        feat = self.ctx_proj(torch.cat([feat_seas, feat_loc], dim=-1))
        q = sigma = None
        if self.prob_head:
            center = mu.detach() if self.prob_detach else mu
            if self.horizon_cond_head:
                fh = feat.unsqueeze(1) + self.horizon_emb.unsqueeze(0)   # (B,H,hidden)
                q = self.quantile_head(fh).view(-1, self.pred_len, nf, self.K)
                sigma = self.scale_head(fh) + 1e-3                       # (B, H, nf)
            else:
                q = self.quantile_head(feat).view(-1, 1, nf, self.K)
                sigma = self.scale_head(feat) + 1e-3                     # (B, nf)
            q = q + center.unsqueeze(-1)                 # (B, H, nf, K)

        # cache the pieces compute_loss needs to rebuild the target in
        # the same de-cycled / anchored stream frame
        self._cyc_fut = (cyc_d + cyc_w) if cyc_d is not None else None
        self._seq_last = seq_last

        if cyc_d is not None:
            if self.cycle_gate:
                steps = torch.arange(1, self.pred_len + 1,
                                     device=cyc_d.device)
                tgate = steps.to(cyc_d.dtype) / self.pred_len
                cyc_d = cyc_d * torch.sigmoid(
                    self.gate_d[0] - self.gate_d[1] * tgate).unsqueeze(-1)
                cyc_w = cyc_w * torch.sigmoid(
                    self.gate_w[0] - self.gate_w[1] * tgate).unsqueeze(-1)
            mu = mu + cyc_d + cyc_w
            if q is not None:
                q = q + (cyc_d + cyc_w).unsqueeze(-1)
        if seq_last is not None:
            mu = mu + seq_last
            if q is not None:
                q = q + seq_last.unsqueeze(-1)

        # RevIN inverse (load-channel stats, valid in both revin scopes)
        win_std = self.revin.stdev[:, :1, :nf]
        win_mean = self.revin.mean[:, :1, :nf]
        mu = mu * win_std + win_mean
        if q is not None:
            q = q * win_std.unsqueeze(-1) + win_mean.unsqueeze(-1)

        out = {'mu': mu}
        if self.physics_identity:
            mu_raw = mu * self.load_scale.view(1, 1, nf)
            out['cmbbtu'] = (mu_raw * (self.cel_coef * self.load_scale)
                             .view(1, 1, nf)).sum(dim=-1, keepdim=True)
        if self.prob_head:
            out['quantiles'], out['sigma'] = q, sigma
        # internal streams exposed for the component-specialist loss
        out['_streams'] = {'seasonal': s, 'trend': t, 'local': loc,
                           'weather': fut}
        return out

    def compute_loss(self, out, y_true, scale3):
        """Decoupled probabilistic training + component-specialist losses.

        Shared paths (identical to PERC-Net):
          l_point (MAPE) — aligns the composed sum with the metric
          l_pin / l_nll / l_cvar / l_mono — distributional specialists
        v2 component L1s (normalized scale, exclusive targets):
          seasonal / trend heads → decomposition of the target
          local stream → residual left by s+t (detached)
          weather branch → residual left by s+t+loc (detached)
        """
        nf = self.n_free
        mu = out['mu']
        streams = out['_streams']
        q = out.get('quantiles')
        y_raw = y_true * scale3

        if q is None:
            o_raw = mu * scale3
            return ((y_raw - o_raw).abs()
                    / (y_raw.abs() + 1e-2)).mean() * 100

        sigma = out['sigma'].clamp(min=1e-3)
        q_raw = q * scale3.view(1, 1, nf, 1)
        mu_raw = mu * scale3
        # sigma is (B, nf) in the v2 head, (B, H, nf) when horizon-conditioned
        s_sc = sigma if sigma.dim() == 3 else sigma.view(-1, 1, nf)

        # shared decoupled paths (identical to PERC-Net)
        u05 = y_raw - mu_raw
        if self.point_loss == 'mape':
            l_point = (u05.abs() / (y_raw.abs() + 1e-2)).mean() * 100
        else:
            l_point = 0.5 * u05.abs().mean()

        u = y_raw.unsqueeze(-1) - q_raw
        taus = self.taus.to(u.device).view(1, 1, 1, -1)
        l_pin = torch.maximum(taus * u, (taus - 1) * u).mean() * self.lam_pin

        resid_abs = (y_raw - mu_raw).abs()
        z = resid_abs.detach() / s_sc if self.prob_detach \
            else resid_abs / s_sc
        l_nll = (z + torch.log(s_sc)).mean()

        l_cvar = self.cvar_t + F.relu(z - self.cvar_t).mean() \
            / (1.0 - self.beta)

        l_mono = F.relu(-torch.diff(q, dim=-1)).mean()

        # ── component-specialist losses ────────────────────────────────
        # rebuild the target in the stream frame.  NOTE units: y_true is
        # column-normalized and win_mean/win_std are window statistics of
        # that same normalized space, so use y_true (NOT y_raw) here.
        #   r_full = (y_true - win_mean)/win_std - cyc_fut - anchor
        win_std = self.revin.stdev[:, :1, :nf]           # (B, 1, nf)
        win_mean = self.revin.mean[:, :1, :nf]
        r_full = (y_true - win_mean) / win_std
        if self._cyc_fut is not None:
            r_full = r_full - self._cyc_fut
        if self._seq_last is not None:
            r_full = r_full - self._seq_last
        r_full = r_full.detach()                         # target, no grad

        s, t, loc = streams['seasonal'], streams['trend'], streams['local']
        seas_tgt, trend_tgt = self.decomp(r_full)
        l_seas = (s - seas_tgt).abs().mean()
        l_trend = (t - trend_tgt).abs().mean()
        lin_out = (s + t).detach()
        l_loc = (loc - (r_full - lin_out)).abs().mean()
        if streams['weather'] is not None:
            prev = (lin_out + loc.detach())
            l_weather = (streams['weather']
                         - (r_full - prev)).abs().mean()
        else:
            l_weather = torch.zeros((), device=mu.device)

        return (l_pin + l_point + self.lam_nll * l_nll
                + self.lam_tail * l_cvar + self.lam_mono * l_mono
                + self.lam_dec * (l_seas + l_trend + l_loc + l_weather))


class ModelMH(Model):
    """Multi-horizon PERC-Net v2: ONE model delivers 24/48/72/96 h.

    Direct multi-horizon setup: the backbone and all heads are sized for
    the maximum horizon (96); a single forward emits the full 96-step
    path, and each horizon h is served by slicing mu[:, :h].  Because the
    24-step forecast of the 96-step path IS the 24h forecast (same window
    anchor), cross-horizon consistency holds by construction.

    Loss: the decoupled probabilistic objective evaluated on each horizon
    slice and averaged — per-horizon point/pinball supervision on shared
    backbone streams.
    """

    HORIZONS = (24, 48, 72, 96)

    def compute_loss_mh(self, out, y_true, scale3,
                        horizons=HORIZONS):
        # component-loss target cache must be sliced to the same horizon
        cyc96 = getattr(self, '_cyc_fut', None)
        total = 0.0
        for h in horizons:
            self._cyc_fut = cyc96[:, :h] if cyc96 is not None else None
            sub = {'mu': out['mu'][:, :h],
                   'quantiles': (out['quantiles'][:, :h]
                                 if out.get('quantiles') is not None
                                 else None),
                   'sigma': (out['sigma'][:, :h]
                             if out.get('sigma') is not None
                             and out['sigma'].dim() == 3
                             else out.get('sigma'))}
            if '_streams' in out:
                sub['_streams'] = {k: (v[:, :h] if v is not None else None)
                                   for k, v in out['_streams'].items()}
            total = total + self.compute_loss(sub, y_true[:, :h], scale3)
        self._cyc_fut = cyc96
        return total / len(horizons)
