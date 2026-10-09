import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


class DataLoaderS(object):
    """Sliding-window data loader for time series forecasting.

    use_calendar: append hour-of-day (sin/cos) and day-of-week (sin/cos)
    derived from absolute row position (hourly data assumed). Columns are
    appended AFTER any diff features, so target columns [:3] are unchanged.
    """

    def __init__(self, file_name, train, valid, device, horizon, window,
                 normalize=2, multi_season_val=True, use_diff=False,
                 use_calendar=False, use_future_weather=False,
                 use_rolling_weather=False, nwp_noise_sigma=0.0,
                 nwp_noise_seed=2024):
        self.P = window
        self.h = horizon
        self.use_diff = use_diff
        self.use_calendar = use_calendar
        self.use_future_weather = use_future_weather
        self.use_rolling_weather = use_rolling_weather
        # NWP realism: forecast-error injection.  nwp_noise_sigma is the
        # per-24h-lead noise std as a fraction of each weather channel's
        # train-period std, growing as sqrt(lead/24) (error growth with
        # forecast lead).  0.0 = perfect-prognosis upper bound.
        self.nwp_noise_sigma = nwp_noise_sigma
        self.nwp_noise_seed = nwp_noise_seed
        fin = open(file_name)
        try:
            df = pd.read_csv(fin)
            num = df.select_dtypes(include=[np.number])
            self.rawdat = num.to_numpy(dtype=np.float64)
            self.columns = list(num.columns)
        except Exception:
            fin.seek(0)
            self.rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
            self.columns = [f'ch{i}' for i in range(self.rawdat.shape[1])]
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self.offset = np.zeros(self.m)

        train_end = int(train * self.n)
        valid_end = int((train + valid) * self.n)
        self.train_end = train_end

        # ── Build train / val / test index sets ──
        if multi_season_val and valid > 0:
            pre_test_n = valid_end
            n_chunks = 4
            chunk_size = pre_test_n // n_chunks
            val_per_chunk = max(1, int(valid * pre_test_n / n_chunks))

            train_idx, val_idx = [], []
            offset = self.P + self.h - 1
            for c in range(n_chunks):
                cs = c * chunk_size
                ce = cs + chunk_size if c < n_chunks - 1 else pre_test_n
                vs = ce - val_per_chunk
                train_idx.extend(range(max(cs, offset), vs))
                val_idx.extend(range(vs, ce))
            test_idx = list(range(pre_test_n, self.n))
        else:
            offset = self.P + self.h - 1
            train_idx = list(range(offset, train_end))
            val_idx = list(range(train_end, valid_end))
            test_idx = list(range(valid_end, self.n))

        # ── Fit normalization on train data only ──
        self._normalized(normalize, train_idx)
        # per-channel train std (raw units) — noise scale reference
        tr = self.rawdat[train_idx, :]
        self.train_sd = tr.std(axis=0) + 1e-8

        # ── Rolling accumulated-weather features (causal, causal means use
        #    only rows <= t; for future rows they roll over forecast weather
        #    — operational NWP statistics, no leakage) ──
        self.roll = None
        if use_rolling_weather:
            self.roll = self._build_rolling([3, 7], [24, 72])

        # ── Create batched datasets ──
        self.train = self._batchify(train_idx, self.h)
        self.valid = self._batchify(val_idx, self.h)
        self.test = self._batchify(test_idx, self.h)

        self.scale = torch.from_numpy(self.scale).float().to(device)
        self.device = device

    def _normalized(self, normalize, train_indices):
        if normalize == 3:
            # z-score (TSLib convention), fit on the train segment only
            train_rawdat = self.rawdat[train_indices, :]
            for i in range(self.m):
                mu = train_rawdat[:, i].mean()
                sd = train_rawdat[:, i].std() + 1e-8
                self.scale[i] = sd
                self.offset[i] = mu
                self.dat[:, i] = (self.rawdat[:, i] - mu) / sd
            return
        if not normalize:
            self.scale = np.ones(self.m)
            self.dat = self.rawdat.copy()
            return
        train_rawdat = self.rawdat[train_indices, :]
        for i in range(self.m):
            col = train_rawdat[:, i]
            abs_col = np.abs(col)
            raw_max = np.max(abs_col)
            p95 = np.percentile(abs_col, 95) if len(abs_col) >= 20 else raw_max
            robust_scale = max(p95 * 2.0, 1e-8)
            if raw_max > robust_scale * 50.0:
                self.scale[i] = robust_scale
            else:
                self.scale[i] = raw_max
            self.scale[i] = max(self.scale[i], 1e-8)
            self.dat[:, i] = np.clip(self.rawdat[:, i] / self.scale[i], -10.0, 10.0)

    def _rolling_mean(self, x, w):
        """Causal rolling mean over window w (expanding at the start)."""
        c = np.cumsum(np.insert(x, 0, 0.0))
        idx = np.arange(len(x))
        lo = np.maximum(idx - w + 1, 0)
        return (c[idx + 1] - c[lo]) / (idx - lo + 1)

    def _build_rolling(self, cols, windows):
        """(n, len(cols)×len(windows)) rolling means of normalized dat.
        cols: column indices (3=TEMP, 7=WBT)."""
        feats = [self._rolling_mean(self.dat[:, c], w)
                 for c in cols for w in windows]
        return np.stack(feats, axis=-1)                     # (n, 4)

    def _calendar_features(self, abs_rows):
        """(N, P) absolute row indices → (N, P, 4) calendar encodings."""
        hod = (abs_rows % 24) / 24.0 * 2 * np.pi
        dow = ((abs_rows % 168) // 24) / 7.0 * 2 * np.pi
        return np.stack([np.sin(hod), np.cos(hod),
                         np.sin(dow), np.cos(dow)], axis=-1)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        idx_set = list(idx_set)
        ends = [idx_set[i] - horizon + 1 for i in range(n)]
        starts = [e - self.P for e in ends]
        X = np.stack([self.dat[s:e, :] for s, e in zip(starts, ends)])  # (N, P, 12)
        Y = np.stack([self.dat[idx_set[i] + 1 - horizon:idx_set[i] + 1, :]
                      for i in range(n)])

        # Inject differential features: x[t] - x[t-1]
        if self.use_diff:
            X_diff = np.diff(X, axis=1, prepend=X[:, :1, :])
            X = np.concatenate([X, X_diff], axis=-1)       # (N, P, 24)

        # Inject calendar features (hour-of-day / day-of-week, row-based)
        if self.use_calendar:
            abs_rows = np.stack([np.arange(s, e)
                                 for s, e in zip(starts, ends)])
            X = np.concatenate([X, self._calendar_features(abs_rows)],
                               axis=-1)

        # Inject rolling accumulated-weather features (chiller thermal mass)
        if self.use_rolling_weather:
            X = np.concatenate(
                [X, np.stack([self.roll[s:e, :] for s, e in
                              zip(starts, ends)])], axis=-1)

        # Append future rows as known covariates (operational NWP setting):
        # LEAKAGE GUARD — zero all target-derived columns.  Besides the three
        # loads themselves, CMBMBBTU (CEL) is an EXACT linear combination of
        # the loads (CEL = 0.0034·KW + 0.0120·CHW + 1.0·HT, R² = 1.000000)
        # and GHG is ~75% predictable from loads; only weather columns
        # 3:10 are legitimate exogenous covariates.
        if self.use_future_weather:
            fut_abs = np.stack([np.arange(e, e + self.h)
                                for e in ends])                # (N, h)
            fut = self.dat[fut_abs].copy()                     # (N, h, m)
            fut[:, :, 0:3] = 0.0                               # loads
            fut[:, :, 10:12] = 0.0                             # CEL / GHG
            if self.nwp_noise_sigma > 0.0:
                # forecast-error injection on weather cols 3:10: std grows
                # with lead √(lead/24), scaled by each channel's train std,
                # converted to normalized units.  Deterministic per
                # (seed, window-end) so every run reproduces exactly.
                leads = np.arange(1, self.h + 1, dtype=np.float64)
                growth = np.sqrt(leads / 24.0)                 # (h,)
                sdn = (self.train_sd[3:10] / self.scale[3:10]
                       ).astype(np.float64)                    # (w,) norm units
                sig = self.nwp_noise_sigma * sdn[None, :] * growth[:, None]
                for i in range(fut.shape[0]):
                    rng = np.random.default_rng(
                        self.nwp_noise_seed * 1000003 + int(ends[i]))
                    fut[i, :, 3:10] += rng.normal(
                        0.0, sig).astype(fut.dtype)
            if self.use_calendar:
                fut = np.concatenate(
                    [fut, self._calendar_features(fut_abs)], axis=-1)
            if self.use_rolling_weather:
                fut = np.concatenate(
                    [fut, self.roll[fut_abs]], axis=-1)
            X = np.concatenate([X, fut], axis=1)               # (N, P+h, C)

        return [torch.from_numpy(X).float(), torch.from_numpy(Y).float()]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.arange(length)
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt].to(self.device)
            Y = targets[excerpt].to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size
