#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main comparison table: PERC-Net against the baseline families.

Trains ``model x horizon x seed`` and appends one CSV row per run to
``results/main_results.csv``.  Runs already present (same
model/setting/horizon/seed) are skipped, so an interrupted sweep can be
re-launched.

Settings
--------
``base``  past 168 h + calendar features, no future covariates
``nwp``   base + the future weather rows as known covariates — the
          operational numerical-weather-prediction setting, and the one
          the paper reports as PERC-Net's main configuration

Usage
-----
    python -m experiments.main_table --models PERCNet --seeds 2024,2025,2026
    python -m experiments.main_table --models DLinear,iTransformer,Mamba,\\
        Crossformer,FiLM,FESTRNN --setting base
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from config import DATA_FILE, PROJECT_ROOT, SEQ_LEN
from percnet.data import DataLoaderS
from percnet import model as percnet_model
from percnet import model_ablation as percnet_abl
from percnet.baselines import MODEL_REGISTRY, count_parameters as count_base
from experiments.common import (MAIN_HEADER, append_csv_row, build_metrics_row,
                                measure_flops)
from experiments.harness import (configure_backends, detect_device, fix_seed,
                                 mps_warmup, train_and_eval, validate_mps_fft)

TARGET_NAMES = ['KW', 'CHWTON', 'HTMBTU']
TRAIN_FRAC, VALID_FRAC = 0.8, 0.1   # test is the remaining 0.1

# Models whose forward consumes the future rows internally (the input
# window is therefore seq_len + pred_len); the others get the future rows
# appended to the window by the loader itself.
FUTURE_AWARE = ('PERCNet', 'PERCNetAbl')


# ─────────────────────────────────────────────────────────────────────────
# Per-dataset presets
# ─────────────────────────────────────────────────────────────────────────
# ASU is the dataset shipped with this repository: it is the only one with
# the CEL physics identity and genuine future weather covariates.  The
# TSLib presets are kept so the loader/model code paths for generic
# M-forecasting data remain exercisable; drop the corresponding public
# CSVs into ``tslib_datasets/`` to use them.
DATASET_PRESETS = {
    'ASU': dict(file=DATA_FILE, enc_in=20, n_free=3, n_tgt=3,
                physics_identity=True, revin_scope='loads',
                use_calendar=True, use_rolling_weather=True,
                seq_len=SEQ_LEN, zscore=False),
    'ETTm1': dict(file='tslib_datasets/ETT-small/ETTm1.csv', enc_in=7,
                  n_free=7, n_tgt=7, physics_identity=False,
                  revin_scope='all', use_calendar=False,
                  use_rolling_weather=False, seq_len=SEQ_LEN, zscore=True),
    'ETTh1': dict(file='tslib_datasets/ETT-small/ETTh1.csv', enc_in=7,
                  n_free=7, n_tgt=7, physics_identity=False,
                  revin_scope='all', use_calendar=False,
                  use_rolling_weather=False, seq_len=SEQ_LEN, zscore=True),
    'ECL': dict(file='tslib_datasets/electricity/electricity.csv', enc_in=321,
                n_free=321, n_tgt=321, physics_identity=False,
                revin_scope='all', use_calendar=False,
                use_rolling_weather=False, seq_len=SEQ_LEN, zscore=True),
    'Exchange': dict(file='tslib_datasets/others/exchange_rate.csv', enc_in=8,
                     n_free=8, n_tgt=8, physics_identity=False,
                     revin_scope='all', use_calendar=False,
                     use_rolling_weather=False, seq_len=SEQ_LEN, zscore=True),
}

# ─────────────────────────────────────────────────────────────────────────
# Training recipes
# ─────────────────────────────────────────────────────────────────────────
# Baselines run under their own published recipes (loss / optimiser /
# schedule) on our unified data pipeline, with honest validation-based
# model selection.  FESTRNN's recipe follows its official repository.
DEFAULT_RECIPE = dict(loss_type='swl1', lr=3e-4, weightdecay=1e-3,
                      clip=0.3, epochs=100, scheduler='cosine')

MODEL_RECIPES = {
    # FESTRNN official: MAPE on the raw scale, Adam(8e-4), no weight decay
    # (the config's 'weightdecay: 0.5' is the plateau factor), grad clip 5,
    # ReduceLROnPlateau(0.5, patience=2) on val MAPE, 50 epochs, batch 64.
    'FESTRNN': dict(loss_type='mape_raw', lr=8e-4, weightdecay=0.0, clip=None,
                    epochs=50, scheduler='plateau', plateau_factor=0.5,
                    plateau_patience=2, early_stop=False),
    # PERC-Net: the objective lives inside the model (pinball + Laplace NLL
    # + RU-CVaR + decomposition-matched component losses).  150 epochs —
    # 50 left the 48-96 h validation curves still descending.
    'PERCNet': dict(loss_type='model', lr=8e-4, weightdecay=0.0, clip=None,
                    epochs=150, scheduler='plateau', plateau_factor=0.5,
                    plateau_patience=6, early_stop=False),
    # Ablative variant of the same model (same recipe, same budget).
    'PERCNetAbl': dict(loss_type='model', lr=8e-4, weightdecay=0.0, clip=None,
                       epochs=150, scheduler='plateau', plateau_factor=0.5,
                       plateau_patience=6, early_stop=False),
}


def build_config(model_name, horizon, setting, overrides=None, dataset='ASU'):
    """Namespace of hyper-parameters for one (model, horizon, setting)."""
    from types import SimpleNamespace
    preset = DATASET_PRESETS[dataset]
    cfg = SimpleNamespace(
        seq_len=preset['seq_len'], pred_len=horizon,
        enc_in=preset['enc_in'], c_out=preset['n_tgt'],
        horizons=[horizon], dataset=dataset,
        use_calendar=preset['use_calendar'],
        use_rolling_weather=preset['use_rolling_weather'],
        n_free=preset['n_free'], n_tgt=preset['n_tgt'],
        physics_identity=preset['physics_identity'],
        revin_scope=preset['revin_scope'],
        use_future_weather=(setting == 'nwp'),
        loss_scale_weighted=True, multi_season_val=False,
        nwp_noise_sigma=0.0, nwp_noise_seed=2024,
        **dict(DEFAULT_RECIPE, batchsize=64,
               weightdecay=DEFAULT_RECIPE['weightdecay']))
    for k, v in MODEL_RECIPES.get(model_name, {}).items():
        setattr(cfg, k, v)

    # ── PERC-Net, paper configuration ──────────────────────────────────
    # These values are exactly the ones baked into percnet/model.py; they
    # are restated here so the ablative variant starts from the same point
    # and so model_ablation can reproduce the paper model bit-for-bit.
    if model_name in ('PERCNet', 'PERCNetAbl'):
        cfg.n_quantiles = 9
        cfg.cvar_beta = 0.9
        cfg.lam_pin = 1.0
        cfg.lam_nll = 1.0
        cfg.lam_tail = 0.5
        cfg.lam_mono = 0.1
        cfg.lam_dec = 10.0
        cfg.linear_hidden = 36
        cfg.decomp_kernel = 13
        cfg.prnn_seg_len = 24
        cfg.prnn_dim = 12
        cfg.prnn_dropout = 0.1
        cfg.dropout = 0.1
        cfg.use_last_value_anchor = True
        cfg.use_cycle_queue = True
        cfg.prob_head = True
        cfg.prob_detach = False          # joint: probabilistic loss reaches mu
        cfg.horizon_cond_head = True
        cfg.cycle_gate = False
        cfg.point_loss = 'mape'
        cfg.revin_scope = 'loads'
    elif model_name == 'DLinear':
        cfg.individual, cfg.moving_avg = True, 25
    elif model_name == 'iTransformer':
        cfg.d_model, cfg.n_heads = 128, 8
        cfg.n_layers, cfg.d_ff = 2, 256
    elif model_name == 'Mamba':
        cfg.d_model, cfg.n_layers, cfg.d_state = 64, 2, 16
    elif model_name == 'Crossformer':
        cfg.seg_count, cfg.d_model, cfg.n_heads, cfg.n_layers = 6, 64, 4, 2
    elif model_name == 'FiLM':
        cfg.degree, cfg.period = 24, 24
    elif model_name == 'FESTRNN':
        cfg.hidden_size, cfg.d_model = 64, 16
        cfg.seg_len, cfg.decomp_kernel = 24, 13
        cfg.dropout = 0.4

    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)

    # Calendar + rolling-weather feature columns are appended by the loader
    # for ASU; recompute the effective channel count.
    if cfg.dataset == 'ASU':
        cfg.enc_in = (12 + (4 if cfg.use_calendar else 0)
                      + (4 if cfg.use_rolling_weather else 0))
    cfg.seq_len = SEQ_LEN

    # Future-aware models read the future rows inside forward(), so their
    # window is seq_len + pred_len.  Everything else gets the future rows
    # appended to the window by the loader, which is what the baseline
    # interface expects.
    if setting == 'nwp' and model_name not in FUTURE_AWARE:
        cfg.seq_len = SEQ_LEN + horizon
    return cfg


def instantiate(model_name, cfg, device, scale_arr, dataset):
    """Build the model for ``model_name`` and report its parameter count."""
    if model_name == 'PERCNet':
        # The paper model: a frozen, self-contained file with an explicit
        # keyword interface (no config object).
        model = percnet_model.Model(
            seq_len=cfg.seq_len, pred_len=cfg.pred_len, enc_in=cfg.enc_in,
            n_free=cfg.n_free, hidden=cfg.linear_hidden,
            decomp_kernel=cfg.decomp_kernel, seg_len=cfg.prnn_seg_len,
            d_model=cfg.prnn_dim, dropout=cfg.prnn_dropout,
            n_quantiles=cfg.n_quantiles, beta=cfg.cvar_beta,
            lam_pin=cfg.lam_pin, lam_nll=cfg.lam_nll, lam_tail=cfg.lam_tail,
            lam_mono=cfg.lam_mono, lam_dec=cfg.lam_dec).to(device)
        params = sum(p.numel() for p in model.parameters())
    elif model_name == 'PERCNetAbl':
        model = percnet_abl.Model(cfg).to(device)
        params = sum(p.numel() for p in model.parameters())
    else:
        model = MODEL_REGISTRY[model_name](cfg).to(device)
        params, _ = count_base(model)

    # The CEL identity layer needs the per-channel data scale to emit the
    # output in physical units.
    if dataset == 'ASU' and hasattr(model, 'load_scale'):
        model.load_scale.copy_(torch.tensor(
            scale_arr[:cfg.n_free].astype(np.float32), device=device))
    return model, params


def run_one(model_name, setting, horizon, seed, device, out_dir,
            overrides=None, dataset='ASU'):
    """Train + evaluate one (model, setting, horizon, seed).  Returns a row."""
    os.makedirs(out_dir, exist_ok=True)
    preset = DATASET_PRESETS[dataset]

    if dataset != 'ASU' and setting == 'nwp':
        setting = 'base'    # generic datasets carry no exogenous future rows

    cfg = build_config(model_name, horizon, setting, overrides, dataset)
    data = DataLoaderS(preset['file'], TRAIN_FRAC, VALID_FRAC, device,
                       horizon, preset['seq_len'],
                       normalize=(3 if preset.get('zscore') else 2),
                       multi_season_val=cfg.multi_season_val,
                       use_calendar=cfg.use_calendar,
                       use_future_weather=cfg.use_future_weather,
                       use_rolling_weather=cfg.use_rolling_weather,
                       nwp_noise_sigma=getattr(cfg, 'nwp_noise_sigma', 0.0),
                       nwp_noise_seed=getattr(cfg, 'nwp_noise_seed', 2024))
    scale_arr = data.scale.cpu().numpy()
    offset_arr = getattr(data, 'offset', np.zeros_like(scale_arr))

    fix_seed(seed)
    model, params = instantiate(model_name, cfg, device, scale_arr, dataset)

    # FLOPs: the future-aware models take seq_len + pred_len input rows.
    n_rows = cfg.seq_len + (cfg.pred_len if model_name in FUTURE_AWARE else 0)
    flops = measure_flops(model, n_rows, cfg.enc_in)
    mps_warmup(model, device, cfg.seq_len, cfg.enc_in)

    channel_w = None
    if cfg.loss_scale_weighted and cfg.loss_type != 'model':
        s3 = scale_arr[:cfg.n_tgt]
        channel_w = torch.tensor((s3 / s3.mean()).astype(np.float32),
                                 device=device).view(1, 1, cfg.n_tgt)

    batch = cfg.batchsize
    while True:
        try:
            total_m, per_t, train_time, infer_time = train_and_eval(
                model, data, cfg, channel_w, device, scale_arr, offset_arr,
                out_dir, batch, zscore=preset.get('zscore', False))
            break
        except torch.cuda.OutOfMemoryError:
            if batch <= 8:
                raise
            batch //= 2
            torch.cuda.empty_cache()
            print(f'  [OOM] batch → {batch}, retrying', flush=True)

    return build_metrics_row(model_name, setting, horizon, seed, params,
                             flops, train_time, infer_time, total_m, per_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=str, default='PERCNet')
    ap.add_argument('--horizons', type=str, default='24,48,72,96')
    ap.add_argument('--seeds', type=str, default='2024,2025,2026,2027,2028')
    ap.add_argument('--setting', type=str, default='nwp',
                    choices=['base', 'nwp'])
    ap.add_argument('--dataset', type=str, default='ASU',
                    choices=list(DATASET_PRESETS.keys()))
    ap.add_argument('--device', default=None, choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--csv', default=os.path.join(PROJECT_ROOT, 'results',
                                                  'main_results.csv'))
    args = ap.parse_args()

    device = validate_mps_fft(detect_device(args.device))
    configure_backends(device)
    models = args.models.split(',')
    horizons = [int(h) for h in args.horizons.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'Main table: models={models} horizons={horizons} seeds={seeds} '
          f'setting={args.setting} dataset={args.dataset} device={device}')
    for model_name in models:
        for horizon in horizons:
            for seed in seeds:
                print(f'>>> {model_name} | {args.setting} | h={horizon} '
                      f'| seed={seed}', flush=True)
                out_dir = os.path.join(PROJECT_ROOT, 'results',
                                       f'{model_name}_{args.dataset}'
                                       f'_h{horizon}_s{seed}')
                try:
                    row = run_one(model_name, args.setting, horizon, seed,
                                  device, out_dir, dataset=args.dataset)
                    append_csv_row(args.csv, MAIN_HEADER, row)
                    print(f'  MAE {row["MAE"]:.2f}  MAPE {row["MAPE"]:.2f}%  '
                          f'RMSE {row["RMSE"]:.1f}  '
                          f'(KW {row["MAE_KW"]:.1f} / CHW {row["MAE_CHW"]:.1f}'
                          f' / HT {row["MAE_HT"]:.1f})', flush=True)
                except Exception as e:
                    print(f'  [FAIL] {model_name} h{horizon} s{seed}: {e}',
                          flush=True)
    print(f'\nResults → {args.csv}')


if __name__ == '__main__':
    main()
