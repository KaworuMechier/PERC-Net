#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — paper figures from saved experiment artifacts
================================================================
Reads (per results root):
  main_results.csv                  → horizon bar chart (mean±std over seeds)
  <run_dir>/predictions.npz         → fitting curves (3 targets)
  <run_dir>/quantiles.npz           → quantile coverage + interval width
  <run_dir>/diagnostics.npz         → cycle queues / future gate / kernels

Usage:
  python -m experiments.make_figures --figs results/figs
  python -m experiments.make_figures --models PERCNet,FESTRNN,DLinear
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGETS = ['KW', 'CHW', 'HT']
COLORS = {'PERCNet': '#2196F3', 'FESTRNN': '#FF5722', 'DLinear': '#4CAF50',
          'iTransformer': '#FF9800', 'Mamba': '#9C27B0',
          'Crossformer': '#00BCD4', 'FiLM': '#795548'}


def fig_horizon_bars(df, models, out):
    """Main-table bar chart: MAE mean±std per horizon per model (nwp rows)."""
    horizons = sorted(df.horizon.unique())
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        sub = df[(df.model == m) & (df.setting == 'nwp')]
        if not len(sub):
            sub = df[df.model == m]
        means, stds = [], []
        for h in horizons:
            v = sub[sub.horizon == h].MAE
            means.append(v.mean() if len(v) else np.nan)
            stds.append(v.std() if len(v) > 1 else 0.0)
        ax.bar(np.arange(len(horizons)) + i * width, means, width * 0.9,
               yerr=stds, capsize=3, label=m,
               color=COLORS.get(m, f'C{i}'), edgecolor='k', linewidth=0.5)
    ax.set_xticks(np.arange(len(horizons)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f'{h}h' for h in horizons])
    ax.set_ylabel('MAE (raw units)')
    ax.set_title('Multi-energy load forecasting — MAE by horizon (mean±std)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_horizon_bars.png'), dpi=200)
    plt.close(fig)


def fig_fitting(run_dir, out, n_show=300):
    npz_p = os.path.join(run_dir, 'predictions.npz')
    if not os.path.exists(npz_p):
        return
    z = np.load(npz_p)
    yt, yp = z['y_true'][:n_show], z['y_pred'][:n_show]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, TARGETS)):
        ax.plot(yt[:, :, i].reshape(-1), 'b-', lw=0.7, label='True')
        ax.plot(yp[:, :, i].reshape(-1), 'r--', lw=0.7, label='Pred')
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'fig_fitting_{os.path.basename(run_dir)}.png'),
                dpi=150)
    plt.close(fig)


def fig_coverage(run_dir, out):
    npz_p = os.path.join(run_dir, 'quantiles.npz')
    if not os.path.exists(npz_p):
        return
    z = np.load(npz_p)
    Q, y, taus = z['quantiles'], z['y_true'], z['taus']   # (N,H,3,K)
    nom = np.asarray(taus)
    emp = np.mean(y[..., None] <= Q, axis=(0, 1, 2))       # per-quantile coverage
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
    axes[0].plot(nom, emp, 'o-', color='#2196F3')
    axes[0].set_xlabel('Nominal quantile')
    axes[0].set_ylabel('Empirical coverage')
    axes[0].set_title('Coverage calibration')
    axes[0].grid(alpha=0.3)
    # interval width per horizon step (central 90%)
    lo_i, hi_i = 0, len(taus) - 1
    width = (Q[..., hi_i] - Q[..., lo_i]).mean(axis=(0, 2))
    axes[1].plot(width, 'o-', color='#FF5722')
    axes[1].set_xlabel('Forecast step')
    axes[1].set_ylabel('90% interval width (raw)')
    axes[1].set_title('Interval width vs lead time')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'fig_coverage_{os.path.basename(run_dir)}.png'),
                dpi=150)
    plt.close(fig)


def fig_cycles(run_dir, out):
    p = os.path.join(run_dir, 'diagnostics.npz')
    if not os.path.exists(p) or 'cycle_weekly' not in np.load(p):
        return
    z = np.load(p)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for i, name in enumerate(TARGETS):
        axes[0].plot(z['cycle_daily'][:, i], label=name)
        axes[1].plot(z['cycle_weekly'][:, i], label=name)
    axes[0].set_title('Learned daily cycle (24h)')
    axes[1].set_title('Learned weekly cycle (168h)')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'fig_cycles_{os.path.basename(run_dir)}.png'),
                dpi=150)
    plt.close(fig)


def fig_kernels(run_dir, out):
    p = os.path.join(run_dir, 'diagnostics.npz')
    if not os.path.exists(p) or 'kernels' not in np.load(p):
        return
    z = np.load(p)
    K = z['kernels']                       # (n_exog, 3, kernel_len)
    n_exog, c_out, klen = K.shape
    fig, axes = plt.subplots(c_out, 1, figsize=(12, 2.6 * c_out))
    if c_out == 1:
        axes = [axes]
    for d, ax in enumerate(axes):
        im = ax.imshow(K[:, d, :], aspect='auto', cmap='RdBu_r',
                       vmin=-np.abs(K).max(), vmax=np.abs(K).max())
        ax.set_ylabel(f'exog ch')
        ax.set_title(f'Typed lag kernels → target {TARGETS[d]}')
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[-1].set_xlabel('lag (steps)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'fig_kernels_{os.path.basename(run_dir)}.png'),
                dpi=150)
    plt.close(fig)


def main():
    from config import PROJECT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(PROJECT_ROOT, 'results'))
    ap.add_argument('--figs', default=None)
    ap.add_argument('--models', default='PERCNet,FESTRNN,DLinear,iTransformer,'
                                       'Mamba,Crossformer,FiLM')
    args = ap.parse_args()
    out = args.figs or os.path.join(args.results, 'figs')
    os.makedirs(out, exist_ok=True)

    csv = os.path.join(args.results, 'main_results.csv')
    models = args.models.split(',')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        fig_horizon_bars(df, models, out)
    else:
        print('main_results.csv not found — skip bar chart')

    for run_dir in sorted(glob.glob(os.path.join(args.results, '*_h*_s*'))):
        if not os.path.isdir(run_dir):
            continue
        fig_fitting(run_dir, out)
        fig_coverage(run_dir, out)
        fig_cycles(run_dir, out)
        fig_kernels(run_dir, out)

    print(f'Figures → {out}')


if __name__ == '__main__':
    main()
