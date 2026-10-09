#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multihorizon.py — ONE model, ALL horizons (24/48/72/96)
============================================================
Trains a single multi-horizon PERC-Net once (per seed) on the 96-step
direct path with per-horizon decoupled-probabilistic losses, then
evaluates every horizon by slicing the one forward pass:

  - vs. the per-horizon paradigm (4 separately trained models — every
    baseline in the paper): comparable accuracy at roughly 1/4 the
    training cost, one model to store and deploy instead of four, and
    cross-horizon consistency by construction (the 24 h forecast of the
    96 h path *is* the 24 h path).

Usage:
  python -m experiments.multihorizon --seeds 2024,2025,2026
  python -m experiments.multihorizon --seeds 2024 --cycle_gate
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from config import DATA_FILE, PROJECT_ROOT, SEQ_LEN
from experiments.harness import fix_seed
from percnet.data import DataLoaderS
from percnet.model_ablation import ModelMH
from percnet.metrics import MAE, MAPE, RMSE
from experiments.common import append_csv_row

OUT_CSV = os.path.join(PROJECT_ROOT, 'results', 'multihorizon_results.csv')
HEADER = ['model', 'horizon', 'seed', 'params', 'train_time_s',
          'MAE', 'MAPE', 'RMSE']
HORIZONS = (24, 48, 72, 96)


def evaluate_horizon(model, data, scale_arr, h, batch_size, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, Yb in data.get_batches(data.test[0], data.test[1],
                                       batch_size, False):
            out = model(Xb.to(device))
            preds.append(out['mu'][:, :h, :].cpu())
            trues.append(Yb[:, :h, :3].cpu())
    pred = torch.cat(preds).numpy() * scale_arr[:3]
    true = torch.cat(trues).numpy() * scale_arr[:3]
    return (MAE(true, pred), MAPE(true, pred), RMSE(true, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='2024,2025,2026')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--cycle_gate', action='store_true',
                    help='enable horizon-adaptive cycle gates (v2.1)')
    ap.add_argument('--hw', choices=['invnaive', 'equal'], default='invnaive',
                    help='per-horizon loss weighting: invnaive (default; '
                         'each horizon weighted by 1/persistence-MAE so the '
                         '96h term does not dominate the gradient) or equal')
    ap.add_argument('--device', default=None)
    ap.add_argument('--csv', default=OUT_CSV)
    args = ap.parse_args()

    device = torch.device(args.device or (
        'cuda' if torch.cuda.is_available() else 'cpu'))
    seeds = [int(s) for s in args.seeds.split(',')]

    from types import SimpleNamespace
    for seed in seeds:
        fix_seed(seed)
        # one loader at the max horizon; windows and anchors are shared by
        # every horizon, which is what makes the slice consistent
        data = DataLoaderS(DATA_FILE, 0.8, 0.1, device, 96, SEQ_LEN,
                           normalize=2, multi_season_val=False,
                           use_calendar=True, use_future_weather=True,
                           use_rolling_weather=True)
        scale_arr = data.scale.cpu().numpy()
        scale3 = torch.tensor(scale_arr[:3].astype(np.float32),
                              device=device).view(1, 1, 3)

        cfg = SimpleNamespace(
            seq_len=SEQ_LEN, pred_len=96, enc_in=20, n_free=3, n_tgt=3,
            physics_identity=True, use_calendar=True, use_future_weather=True,
            prob_head=True, prob_detach=False,       # joint = accuracy-first
            point_loss='mape', linear_hidden=36, decomp_kernel=13,
            revin_scope='loads', prnn_seg_len=24, prnn_dim=12,
            prnn_dropout=0.1, lam_dec=10.0, n_quantiles=9, cvar_beta=0.9,
            lam_nll=1.0, lam_tail=0.5, lam_mono=0.1, lam_pin=1.0,
            horizon_cond_head=True,
            cycle_gate=args.cycle_gate, loss_type='model')
        model = ModelMH(cfg).to(device)
        model.load_scale.copy_(torch.tensor(
            scale_arr[:3].astype(np.float32), device=device))
        params = sum(p.numel() for p in model.parameters())

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=6)
        X, Y = data.train

        # per-horizon difficulty reference: persistence (last observed
        # load) MAE on the validation set, raw scale.  Used to (a) build a
        # difficulty-normalized val metric so 96h does not dominate the
        # scheduler, and (b) set inverse-difficulty loss weights.
        model.eval()
        pv, tv = [], []
        with torch.no_grad():
            for Xb, Yb in data.get_batches(data.valid[0],
                                           data.valid[1][:, :, :3],
                                           256, False):
                pv.append(Xb[:, -1, :3].cpu())       # last observed load
                tv.append(Yb.cpu())
        pv = torch.cat(pv).numpy() * scale_arr[:3]
        tv = torch.cat(tv).numpy() * scale_arr[:3]
        naive = np.array([MAE(tv[:, :h], np.repeat(pv[:, None, :], h, axis=1))
                          for h in HORIZONS])
        if args.hw == 'invnaive':
            hw = naive / naive.mean()                # relative difficulty
            hw = torch.tensor(hw / hw.sum() * len(HORIZONS),
                              dtype=torch.float32, device=device)
        else:
            hw = torch.ones(len(HORIZONS), device=device)
        print(f'persistence MAE per horizon: '
              f'{dict(zip(HORIZONS, np.round(naive, 1)))}')
        print(f'loss weights per horizon:   '
              f'{dict(zip(HORIZONS, np.round(hw.cpu().numpy(), 2)))}')

        def val_scores():
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for Xb, Yb in data.get_batches(data.valid[0],
                                               data.valid[1][:, :, :3],
                                               256, False):
                    preds.append(model(Xb)['mu'].cpu())
                    trues.append(Yb.cpu())
            mu = torch.cat(preds).numpy() * scale_arr[:3]
            yt = torch.cat(trues).numpy() * scale_arr[:3]
            maes = np.array([MAE(yt[:, :h], mu[:, :h]) for h in HORIZONS])
            return maes, (maes / naive).mean()       # difficulty-normalized

        best_val, best_state = float('inf'), None
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            for Xb, Yb in data.get_batches(X, Y[:, :, :3], args.batch_size,
                                           True):
                opt.zero_grad()
                out = model(Xb)
                loss = sum(
                    float(hw[i]) * model.compute_loss_mh(
                        out, Yb, scale3, horizons=(h,)) / 1.0
                    for i, h in enumerate(HORIZONS)) / len(HORIZONS)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            maes, val = val_scores()
            sched.step(val)
            if val < best_val:
                best_val = val
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            if ep % 10 == 0 or ep == 1:
                print(f'seed {seed} epoch {ep:3d} | val MAE '
                      f'{dict(zip(HORIZONS, np.round(maes, 1)))} | '
                      f'norm {val:.3f} (best {best_val:.3f})', flush=True)
        train_time = time.time() - t0
        model.load_state_dict(best_state)

        tag = 'PERCNetV2MH' + ('+gate' if args.cycle_gate else '')
        for h in HORIZONS:
            mae, mape, rmse = evaluate_horizon(model, data, scale_arr, h,
                                               args.batch_size, device)
            append_csv_row(args.csv, HEADER,
                           {'model': tag, 'horizon': h, 'seed': seed,
                            'params': params,
                            'train_time_s': round(train_time, 1),
                            'MAE': round(mae, 4), 'MAPE': round(mape, 4),
                            'RMSE': round(rmse, 2)})
            print(f'  seed {seed} h={h}: MAE {mae:.2f} MAPE {mape:.2f}% '
                  f'RMSE {rmse:.1f}', flush=True)
        print(f'  [{tag}] ONE model ({params} params, {train_time:.0f}s) '
              f'serves all 4 horizons', flush=True)


if __name__ == '__main__':
    main()
