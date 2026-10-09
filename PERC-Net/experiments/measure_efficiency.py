#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Efficiency table: parameters, MACs and inference latency of PERC-Net.

Measured on the *paper* model (``percnet.model``) in the NWP setting, at
every horizon, using the same tool and counting convention as the
baselines (``experiments.common.measure_flops``, i.e. torch.profiler with
``with_flops=True``).

MACs are also reported analytically from PyTorch's ``FlopCounterMode``,
which follows the real execution path — this matters because the
lead-time-conditioned probabilistic head runs once per predicted step, so
a "sum of layer shapes" shortcut would under-count it by a factor of H.

Usage
-----
    python -m experiments.measure_efficiency
    python -m experiments.measure_efficiency --device cpu
"""

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from config import DATA_FILE, PROJECT_ROOT, SEQ_LEN
from percnet.data import DataLoaderS
from percnet.model import Model as PERCNet
from experiments.common import measure_flops

ENC_IN = 20                     # 12 raw + 4 calendar + 4 rolling weather


def build(h, device):
    """The paper model at horizon ``h``, with its NWP input window."""
    data = DataLoaderS(DATA_FILE, 0.8, 0.1, device, h, SEQ_LEN, normalize=2,
                       multi_season_val=False, use_calendar=True,
                       use_future_weather=True, use_rolling_weather=True)
    model = PERCNet(seq_len=SEQ_LEN, pred_len=h, enc_in=ENC_IN).to(device)
    model.load_scale.copy_(torch.tensor(
        data.scale.cpu().numpy()[:3].astype(np.float32), device=device))
    return model, data


def macs_flop_counter(model, h, device):
    """Per-sample MACs along the real execution path, in millions."""
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, SEQ_LEN + h, ENC_IN, device=device)
    model.eval()
    with torch.no_grad():
        counter = FlopCounterMode(display=False)
        with counter:
            model(x)
    return counter.get_total_flops() / 2 / 1e6


def measure_latency(model, data, device, n_test=3):
    """Median wall-clock of one full test-set inference pass (batch 64)."""
    times = []
    for _ in range(n_test):
        t0 = time.time()
        with torch.no_grad():
            for Xb, _ in data.get_batches(data.test[0], data.test[1], 64,
                                          shuffle=False):
                model(Xb.to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.time() - t0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None, choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--horizons', type=str, default='24,48,72,96')
    ap.add_argument('--out_csv', default=os.path.join(
        PROJECT_ROOT, 'results', 'efficiency_percnet.csv'))
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    fields = ['model', 'horizon', 'params', 'macs_MMac',
              'flops_MMac', 'inference_time_s']
    rows = []
    for h in [int(x) for x in args.horizons.split(',')]:
        model, data = build(h, device)
        params = sum(p.numel() for p in model.parameters())
        macs = macs_flop_counter(model, h, device)
        flops = measure_flops(model, SEQ_LEN + h, ENC_IN)
        lat = measure_latency(model, data, device)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        rows.append({'model': 'PERC-Net', 'horizon': h, 'params': params,
                     'macs_MMac': round(macs, 4),
                     'flops_MMac': round(flops, 4) if flops == flops else 'NaN',
                     'inference_time_s': round(lat, 4)})
        print(f"PERC-Net h={h:2d} | params {params:6,d} | "
              f"MACs {macs:7.4f} M | inference {lat:.3f} s", flush=True)

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'\n{len(rows)} rows -> {args.out_csv}')


if __name__ == '__main__':
    main()
