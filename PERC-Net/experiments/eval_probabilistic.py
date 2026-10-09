#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_probabilistic.py — probabilistic quality metrics from quantiles.npz
==============================================================================
Reads results/<run>/quantiles.npz (raw scale baked in):
  quantiles (B, H, nf, K), y_true (B, H, nf), taus (K,)

Metrics per run (mean over channels unless noted):
  CRPS        2 x mean pinball loss (exact for the quantile grid, quantized)
  PICP_90     coverage of the [tau=0.05, tau=0.95] central interval
  PINAW_90    average width of that interval (sharpness)
  Winkler_90  Winkler score at alpha=0.10
  QCE         quantile calibration error: mean |empirical tau_hat - tau|
  CVaR_90     mean of the worst-10% absolute errors of the median forecast

Writes results/probabilistic_metrics.csv and prints a summary table.

Usage:  python -m experiments.eval_probabilistic [results_dir]
"""

import glob
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, 'results')
OUT_CSV = os.path.join(RESULTS_DIR, 'probabilistic_metrics.csv')


def worst_frac_mean(err, beta=0.9):
    """Mean of the worst (1-beta) fraction of |err| (RU-CVaR estimate)."""
    err = np.abs(err).ravel()
    k = max(1, int(round((1.0 - beta) * err.size)))
    return np.sort(err)[-k:].mean()


def main(results_dir=RESULTS_DIR):
    # accept either a parent directory (results/) or a specific run dir
    # (results/abl_xxx_h24_s2024)
    if os.path.exists(os.path.join(results_dir, 'quantiles.npz')):
        npz_files = [os.path.join(results_dir, 'quantiles.npz')]
    else:
        npz_files = sorted(glob.glob(
            os.path.join(results_dir, '*', 'quantiles.npz')))
    rows = []
    for npz_path in npz_files:
        run = os.path.basename(os.path.dirname(npz_path))
        z = np.load(npz_path)
        q, y, taus = z['quantiles'], z['y_true'], z['taus']
        i_hi = int(np.argmax(taus))                 # tau=0.95
        i_lo = int(np.argmin(taus))                 # tau=0.05
        i50 = int(np.argmin(np.abs(taus - 0.5)))
        q_lo, q_hi, q50 = q[..., i_lo], q[..., i_hi], q[..., i50]

        crps = 2.0 * np.maximum(
            taus * (y[..., None] - q),
            (taus - 1.0) * (y[..., None] - q)).mean()

        alpha = 0.10
        inside = (y >= q_lo) & (y <= q_hi)
        picp = inside.mean()
        pinaw = (q_hi - q_lo).mean()

        # Winkler score (interval width + 2/alpha penalty outside)
        w = (q_hi - q_lo) \
            + (2.0 / alpha) * np.maximum(q_lo - y, 0.0) \
            + (2.0 / alpha) * np.maximum(y - q_hi, 0.0)
        winkler = w.mean()

        # per-level calibration error (empirical vs nominal tau)
        qce = np.mean([np.abs((y < q[..., i]).mean() - t)
                       for i, t in enumerate(taus)])

        cvar90 = worst_frac_mean(y - q50, beta=0.9)

        rows.append((run, crps, picp, pinaw, winkler, qce, cvar90))
        print(f'{run:44s} CRPS={crps:9.3f}  PICP90={picp:6.3f}  '
              f'PINAW={pinaw:9.2f}  Wink={winkler:9.2f}  '
              f'QCE={qce:6.4f}  CVaR90={cvar90:9.2f}')

    if not rows:
        print(f'no quantiles.npz under {results_dir}')
        return
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    # append with dedup-by-run so successive arm comparisons accumulate
    existing = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            header = f.readline().strip()
            for line in f:
                parts = line.strip().split(',')
                if parts and parts[0]:
                    existing[parts[0]] = line
    for r in rows:
        existing[r[0]] = (f'{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},'
                          f'{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n')
    with open(OUT_CSV, 'w') as f:
        f.write('run,CRPS,PICP_90,PINAW_90,Winkler_90,QCE,CVaR_90\n')
        for line in existing.values():
            f.write(line)
    print(f'\n{len(rows)} new / {len(existing)} total runs → {OUT_CSV}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR)
