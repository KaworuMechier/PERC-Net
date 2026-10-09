"""Plan-B evaluation: tau=0.5 quantile as the point forecast.

Reads results/<run>/quantiles.npz (raw scale already baked in):
  quantiles (B, H, nf, K), y_true (B, H, nf), taus (K,)
Compares MAE of:
  - mu   (= predictions.npz y_pred, the model's point output)
  - q50  (= quantiles[..., tau==0.5], index 4 of the 9-tau grid)
Also reports mean pinball loss (= CRPS/2 estimate) per run.

Usage:  python -m experiments.eval_q50_point [results_dir]
"""

import glob
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, 'results')

Q50_IDX = 4  # taus = linspace(0.05, 0.95, 9) -> [0.05,.15,.25,.35,.45,.5,...]


def pinball(y, q, taus):
    u = y[..., None] - q                       # (B,H,nf,K)
    return np.maximum(taus * u, (taus - 1) * u)


def main(results_dir=RESULTS_DIR):
    rows = []
    for npz_path in sorted(glob.glob(
            os.path.join(results_dir, '*', 'quantiles.npz'))):
        run = os.path.basename(os.path.dirname(npz_path))
        z = np.load(npz_path)
        q, y, taus = z['quantiles'], z['y_true'], z['taus']
        q50 = q[..., Q50_IDX]
        mae_q50 = np.abs(q50 - y).mean()

        pred_path = os.path.join(os.path.dirname(npz_path),
                                 'predictions.npz')
        mae_mu = np.nan
        if os.path.exists(pred_path):
            p = np.load(pred_path)
            if 'y_pred' in p:
                mae_mu = np.abs(p['y_pred'] - y).mean()

        crps = 2.0 * pinball(y, q, taus).mean()
        rows.append((run, mae_mu, mae_q50, crps))
        print(f'{run:48s}  mu_MAE={mae_mu:10.4f}  '
              f'q50_MAE={mae_q50:10.4f}  d={mae_q50 - mae_mu:+9.4f}  '
              f'CRPS~{crps:10.4f}')

    if not rows:
        print(f'no quantiles.npz found under {results_dir}')
        return
    mu = np.mean([r[1] for r in rows])
    q50 = np.mean([r[2] for r in rows])
    print('-' * 100)
    print(f'{"MEAN":48s}  mu_MAE={mu:10.4f}  q50_MAE={q50:10.4f}  '
          f'd={q50 - mu:+9.4f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR)
