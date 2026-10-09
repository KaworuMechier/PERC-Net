#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Component ablations.

Every arm toggles exactly one switch away from the paper configuration, so
the difference against ``full`` is attributable to that switch alone.  Rows
append to ``results/ablation_results.csv``; runs already present
(ablation/horizon/seed) are skipped.

``full`` instantiates the frozen paper model (``percnet.model``).  The
remaining arms use the config-driven variant
(``percnet.model_ablation``); with the paper configuration that variant is
provably identical to the frozen model — same parameter count and same
forward output — which the smoke test checks.

Usage
-----
    python -m experiments.ablation --horizons 24 --seeds 2024,2025,2026
    python -m experiments.ablation --ablations no_nwp,no_cycle
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from config import PROJECT_ROOT
from experiments.common import ABLATION_HEADER, append_csv_row
from experiments.harness import (configure_backends, detect_device,
                                 validate_mps_fft)
from experiments.main_table import DATASET_PRESETS, run_one

# Switch semantics (all arms otherwise identical to the paper model):
#   no_nwp            drop the future-weather branch (base setting)
#   no_cycle          drop the learnable daily/weekly cycle queues
#   no_anchor         drop the last-value anchor (no deviation modelling)
#   revin_all         window-normalise every channel, not just the loads
#   no_nll            Laplace-NLL term off (scale head unsupervised)
#   no_cvar           RU-CVaR tail term off
#   no_component      component (exclusive-target) losses off (lam_dec = 0)
#   no_mono           quantile-monotonicity penalty off
#   point_only        no probabilistic head — point forecast only
#   no_horizon_cond   head feature pooled over the horizon (no lead-time axis)
#   decoupled_probs   probabilistic loss detached from the point path —
#                     isolates the paper's joint-training claim
ABLATIONS = {
    'full':            {'_model': 'PERCNet'},
    'no_nwp':          {'_model': 'PERCNetAbl', 'use_future_weather': False,
                        '_setting': 'base'},
    'no_cycle':        {'_model': 'PERCNetAbl', 'use_cycle_queue': False},
    'no_anchor':       {'_model': 'PERCNetAbl', 'use_last_value_anchor': False},
    'revin_all':       {'_model': 'PERCNetAbl', 'revin_scope': 'all'},
    'no_nll':          {'_model': 'PERCNetAbl', 'lam_nll': 0.0},
    'no_cvar':         {'_model': 'PERCNetAbl', 'lam_tail': 0.0},
    'no_component':    {'_model': 'PERCNetAbl', 'lam_dec': 0.0},
    'no_mono':         {'_model': 'PERCNetAbl', 'lam_mono': 0.0},
    'point_only':      {'_model': 'PERCNetAbl', 'prob_head': False},
    'no_horizon_cond': {'_model': 'PERCNetAbl', 'horizon_cond_head': False},
    'decoupled_probs': {'_model': 'PERCNetAbl', 'prob_detach': True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablations', type=str, default='full,no_nwp,no_cycle,'
                    'no_anchor,revin_all,no_nll,no_cvar,no_component,no_mono,'
                    'point_only,no_horizon_cond,decoupled_probs')
    ap.add_argument('--horizons', type=str, default='24')
    ap.add_argument('--seeds', type=str, default='2024,2025,2026')
    ap.add_argument('--dataset', type=str, default='ASU',
                    choices=list(DATASET_PRESETS.keys()))
    ap.add_argument('--device', default=None, choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--csv', default=os.path.join(PROJECT_ROOT, 'results',
                                                  'ablation_results.csv'))
    args = ap.parse_args()

    device = validate_mps_fft(detect_device(args.device))
    configure_backends(device)
    ablations = args.ablations.split(',')
    horizons = [int(h) for h in args.horizons.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'Ablations: {ablations} horizons={horizons} seeds={seeds} '
          f'dataset={args.dataset} device={device}')
    for name in ablations:
        if name not in ABLATIONS:
            print(f'[SKIP] unknown ablation: {name}')
            continue
        overrides = dict(ABLATIONS[name])
        setting = overrides.pop('_setting', 'nwp')
        model_name = overrides.pop('_model')

        # Raw-scale MAPE is only meaningful for max-normalised data (ASU);
        # a z-scored dataset is zero-centred and MAPE explodes on the
        # near-zero targets.  Fall back to the L1 point loss there.
        if args.dataset != 'ASU':
            overrides.setdefault('point_loss', 'l1')

        for horizon in horizons:
            for seed in seeds:
                print(f'>>> ablation={name} | h={horizon} | seed={seed}',
                      flush=True)
                out_dir = os.path.join(PROJECT_ROOT, 'results',
                                       f'abl_{name}_h{horizon}_s{seed}')
                try:
                    row = run_one(model_name, setting, horizon, seed, device,
                                  out_dir, overrides=overrides,
                                  dataset=args.dataset)
                    append_csv_row(args.csv, ABLATION_HEADER,
                                   {'ablation': name, **row})
                    print(f'  MAE {row["MAE"]:.2f}  MAPE {row["MAPE"]:.2f}%  '
                          f'RMSE {row["RMSE"]:.1f}', flush=True)
                except Exception as e:
                    print(f'  [FAIL] {name} h{horizon} s{seed}: {e}',
                          flush=True)
    print(f'\nAblation results → {args.csv}')


if __name__ == '__main__':
    main()
