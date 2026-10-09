#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nwp_noise.py — NWP forecast-error sensitivity sweep
=======================================================
Injects synthetic forecast error into the future-weather rows (see
DataLoaderS ``nwp_noise_sigma``) and measures how the NWP gain degrades
with forecast quality:

  sigma = 0.0   perfect prognosis (observed weather) — upper bound
  sigma = 0.5   optimistic operational NWP (~0.5σ of climate at 24 h lead)
  sigma = 1.0   pessimistic operational NWP (~GFS-like error at 24 h lead)

Noise std grows with lead time as sqrt(lead/24) and is scaled by each
weather channel's train-period std.  It is deterministic per window end,
so every run reproduces exactly.

Rows append to --csv (cached on (model, setting, horizon, seed)).

Usage:
  python -m experiments.nwp_noise
  python -m experiments.nwp_noise --sigmas pp,nwp_pess
"""

import argparse
import os
import sys
import warnings


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from config import PROJECT_ROOT
from experiments.harness import (configure_backends, detect_device,
                                 validate_mps_fft)
from experiments.main_table import run_one
from experiments.common import MAIN_HEADER, append_csv_row

SIGMAS = {'pp': 0.0, 'nwp_opt': 0.5, 'nwp_pess': 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=str, default='PERCNet')
    ap.add_argument('--horizons', type=str, default='24,48,72,96')
    ap.add_argument('--seeds', type=str, default='2024,2025,2026')
    ap.add_argument('--sigmas', type=str, default='pp,nwp_opt,nwp_pess',
                    help=f'noise levels, keys of {SIGMAS}')
    ap.add_argument('--device', default=None, choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--csv', default=os.path.join(
        PROJECT_ROOT, 'results', 'nwp_noise_sensitivity.csv'))
    args = ap.parse_args()

    device = validate_mps_fft(detect_device(args.device))
    configure_backends(device)
    models = args.models.split(',')
    horizons = [int(h) for h in args.horizons.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    sigmas = args.sigmas.split(',')

    print(f'NWP noise sweep: models={models} sigmas={sigmas} '
          f'horizons={horizons} seeds={seeds}')
    for model_name in models:
        for skey in sigmas:
            sigma = SIGMAS[skey]
            for horizon in horizons:
                for seed in seeds:
                    tag = f'{model_name}_s{skey}_h{horizon}_s{seed}'
                    out_dir = os.path.join(PROJECT_ROOT, 'results',
                                           f'nwpnoise_{tag}')
                    print(f'>>> {tag}', flush=True)
                    try:
                        row = run_one(
                            model_name, 'nwp', horizon, seed, device,
                            out_dir,
                            overrides={'nwp_noise_sigma': sigma,
                                       'nwp_noise_seed': 2024},
                            dataset='ASU')
                        row['model'] = f'{model_name}+{skey}'
                        append_csv_row(args.csv, MAIN_HEADER, row)
                        print(f'  MAE {row["MAE"]:.2f} MAPE {row["MAPE"]:.2f}%',
                              flush=True)
                    except Exception as e:
                        print(f'  [FAIL] {tag}: {e}', flush=True)
    print(f'\nSensitivity results → {args.csv}')


if __name__ == '__main__':
    main()
