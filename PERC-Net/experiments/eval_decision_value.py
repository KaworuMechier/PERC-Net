#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision value of calibrated quantiles under asymmetric cost.

Closes the loop *probabilistic output -> quantile selection -> dispatch
cost* without retraining: it consumes two configurations' artifacts and
evaluates a reserve-commitment (newsvendor) cost.

Decision model
--------------
For one origin and one future step a commitment ``Q`` is made before the
load ``L`` is realised.  Under-supply costs ``c_u`` per unit, over-supply
``c_o`` per unit:

    C(Q) = c_u * max(L - Q, 0) + c_o * max(Q - L, 0)

With ``c_o`` normalised to 1, the minimiser is the critical quantile
``tau* = c_u / (c_u + c_o)`` of the predictive distribution.  Strategies
compared:

    point        Q = the point-only configuration's prediction
    persistence  Q = last observed value
    lag168       Q = same hour of the previous week
    median       Q = tau = 0.50 of the joint configuration (its delivered
                 point forecast, and the baseline the paper reports against)
    quantile     Q = tau* of the joint configuration (linear interpolation)
    oracle       Q = L — the zero-cost lower bound

Inputs
------
    results/PERCNet_ASU_h{h}_s{seed}/    quantiles.npz, predictions.npz
    results/abl_point_only_h{h}_s{seed}/ predictions.npz

Usage
-----
    python -m experiments.main_table --models PERCNet --setting nwp
    python -m experiments.ablation  --ablations point_only
    python -m experiments.eval_decision_value --root results
"""

import argparse
import csv
import os

import numpy as np

HORIZONS = (24, 48, 72, 96)
SEEDS = (2024, 2025, 2026)
RATIOS = (1 / 9, 1 / 3, 1.0, 3.0, 9.0, 19.0)   # c_u / c_o
LOADS = ('Electrical', 'Cooling', 'Heating')
CSV_COLS = ('KW', 'CHWTON', 'HTMBTU')          # dataset_input.csv column order

# Run-directory tags, as written by main_table.py / ablation.py.
JOINT_TAG = 'PERCNet_ASU'                      # paper configuration
POINT_TAG = 'abl_point_only'                   # same model, no prob. head

STRATEGIES = ('persistence', 'lag168', 'median', 'quantile')
# Baseline of the reported cost changes.  'median' is the point forecast
# delivered by the joint configuration (tau = 0.5) — the baseline named in
# the paper's decision model.  'point' compares against the point-only
# configuration's own prediction instead.
REFERENCE = 'median'


def _quantile_at(taus, quants, tau):
    """Linear interpolation of the quantile function at a scalar level.

    ``quants`` is (..., K) in ascending tau order; clamps at the endpoints.
    """
    idx = int(np.searchsorted(taus, tau))
    if idx <= 0:
        return quants[..., 0]
    if idx >= len(taus):
        return quants[..., -1]
    lo, hi = taus[idx - 1], taus[idx]
    w = (tau - lo) / (hi - lo)
    return (1.0 - w) * quants[..., idx - 1] + w * quants[..., idx]


def _cost(load, commit, ratio):
    """Mean asymmetric cost (c_o = 1) over all (origin, step, load)."""
    commit = np.maximum(commit, 0.0)          # a commitment cannot be negative
    return float(np.mean(ratio * np.maximum(load - commit, 0.0)
                         + np.maximum(commit - load, 0.0)))


def run(root, data_csv):
    import pandas as pd
    series = pd.read_csv(data_csv)[list(CSV_COLS)].values
    # test split begins after the 80/10 train/valid blocks
    test_start = int(0.9 * len(series))
    rows = []
    for h in HORIZONS:
        for seed in SEEDS:
            jdir = os.path.join(root, f'{JOINT_TAG}_h{h}_s{seed}')
            pdir = os.path.join(root, f'{POINT_TAG}_h{h}_s{seed}')
            jq, jp = (os.path.join(jdir, 'quantiles.npz'),
                      os.path.join(jdir, 'predictions.npz'))
            pp = os.path.join(pdir, 'predictions.npz')
            if not (os.path.exists(jq) and os.path.exists(jp)
                    and os.path.exists(pp)):
                print(f'  [skip] h{h} s{seed}: artifacts missing '
                      f'(run main_table and the point_only ablation first)')
                continue
            J, P, Pnt = np.load(jq), np.load(jp), np.load(pp)
            load = np.asarray(P['y_true'], dtype=np.float64)
            point = np.asarray(Pnt['y_pred'], dtype=np.float64)
            quants = np.asarray(J['quantiles'], dtype=np.float64)
            taus = np.asarray(J['taus'], dtype=np.float64)

            # persistence: the value observed just before the target window
            pers = np.empty_like(load)
            pers[0] = load[0, 0]
            pers[1:] = load[:-1, 0][:, None, :]

            # lag-168 seasonal baseline: the same hour of the previous
            # week.  Leakage-free for every horizon here, because the
            # latest index it touches stays earlier than the origin.
            base = (test_start - h) + np.arange(load.shape[0])
            seas = series[(base[:, None] + np.arange(h)[None, :]) - 168, :]

            for ratio in RATIOS:
                tau = ratio / (ratio + 1.0)
                qs = _quantile_at(taus, quants, tau)
                qm = _quantile_at(taus, quants, 0.5)
                for d, name in enumerate(LOADS):
                    L = load[:, :, d]
                    rows.append(dict(
                        horizon=h, seed=seed, load=name, ratio=ratio, tau=tau,
                        point=_cost(L, point[:, :, d], ratio),
                        persistence=_cost(L, pers[:, :, d], ratio),
                        lag168=_cost(L, seas[:, :, d], ratio),
                        median=_cost(L, qm[:, :, d], ratio),
                        quantile=_cost(L, qs[:, :, d], ratio),
                        oracle=0.0,
                    ))
    return rows


def summarise(rows, ref=REFERENCE):
    """Relative cost change vs ``ref``, per channel, macro-averaged.

    Each row is reduced to a dimensionless ratio (strategy / ref) before
    averaging, so the three load channels — different units, two orders of
    magnitude apart — contribute equally.  This matches the macro-average
    convention used for MAE in the paper.
    """
    out = {}
    for ratio in RATIOS:
        sel = [r for r in rows if r['ratio'] == ratio]
        per_channel, macro = {}, {}
        for name in LOADS:
            ch = [r for r in sel if r['load'] == name]
            per_channel[name] = {
                s: float(np.mean([100.0 * (r[s] - r[ref]) / r[ref]
                                  for r in ch]))
                for s in STRATEGIES}
        for s in STRATEGIES:
            macro[s] = float(np.mean([per_channel[n][s] for n in LOADS]))
        out[ratio] = {'per_channel': per_channel, 'macro': macro}
    return out


def macro_runs(rows, ref=REFERENCE):
    """Per-run macro-averaged cost change (%), for every strategy.

    Returns ``({ratio: {strategy: array}}, keys)`` where ``keys`` is the
    sorted list of ``(horizon, seed)`` runs actually present, and each
    array is aligned to it — the spread reported next to the paper's
    table.  Restricting to the runs that exist keeps a partial sweep
    readable instead of silently averaging in NaNs.
    """
    keys = sorted({(r['horizon'], r['seed']) for r in rows})
    out = {}
    for ratio in RATIOS:
        sel = [r for r in rows if r['ratio'] == ratio]
        per = {}
        for s in STRATEGIES:
            vals = []
            for h, seed in keys:
                ch = [float(np.mean([100.0 * (r[s] - r[ref]) / r[ref]
                                     for r in sel
                                     if r['horizon'] == h and r['seed'] == seed
                                     and r['load'] == n]))
                      for n in LOADS]
                vals.append(float(np.mean(ch)))
            per[s] = np.array(vals)
        out[ratio] = per
    return out, keys


def main():
    from config import DATA_FILE, PROJECT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(PROJECT_ROOT, 'results'))
    ap.add_argument('--data', default=DATA_FILE)
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'results',
                                                  'decision_value.csv'))
    ap.add_argument('--reference', choices=('median', 'point'),
                    default=REFERENCE,
                    help='baseline of the reported cost changes '
                         '(default: %(default)s)')
    args = ap.parse_args()

    rows = run(args.root, args.data)
    if not rows:
        raise SystemExit(f'no artifacts found under {args.root}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    S = summarise(rows, args.reference)
    print(f'\nreference strategy = {args.reference} '
          f'(median = the tau=0.5 point forecast delivered by the joint config)')
    print('\nA. critical-quantile strategy: expected cost change vs reference, '
          'per channel')
    print(f'{"c_u/c_o":>8} {"tau*":>6} | '
          + ' | '.join(f'{n:>10}' for n in LOADS) + ' | macro')
    print('-' * 62)
    for ratio in RATIOS:
        pc = S[ratio]['per_channel']
        print(f'{ratio:>8.3g} {ratio / (ratio + 1):>6.2f} | '
              + ' | '.join(f'{pc[n]["quantile"]:>+9.1f}%' for n in LOADS)
              + f' | {S[ratio]["macro"]["quantile"]:>+8.1f}%')

    print('\nB. macro-averaged cost change per strategy')
    print(f'{"c_u/c_o":>8} {"tau*":>6} | '
          + ' | '.join(f'{s:>12}' for s in STRATEGIES))
    print('-' * 58)
    for ratio in RATIOS:
        m = S[ratio]['macro']
        print(f'{ratio:>8.3g} {ratio / (ratio + 1):>6.2f} | '
              + ' | '.join(f'{m[s]:>+11.1f}%' for s in STRATEGIES))
    print('\n(negative = cheaper. c_o normalised to 1; each row is '
          'normalised before averaging.)')

    R, keys = macro_runs(rows)
    horizons = sorted({h for h, _ in keys})
    print(f'\nC. dispersion across {len(keys)} runs (ddof=0; '
          f'{len(horizons)} horizons x {len(keys) // max(len(horizons), 1)} '
          f'seeds per horizon)')
    print('   NOTE: this is not a confidence interval — each run value is '
          'itself a mean over 10^5-10^6 samples.')
    print(f'{"c_u/c_o":>8} {"tau*":>6} | '
          + ' | '.join(f'{s:>12}' for s in STRATEGIES))
    for ratio in RATIOS:
        print(f'{ratio:>8.3g} {ratio / (ratio + 1):>6.2f} | '
              + ' | '.join(f'{R[ratio][s].std():>12.3f}' for s in STRATEGIES))

    print('\nD. per-horizon breakdown (within-horizon mean / seed sd, ddof=0), '
          f'column order H = {"/".join(str(h) for h in horizons)}')
    for ratio in RATIOS:
        print(f'  c_u/c_o = {ratio:.3g}   tau* = {ratio / (ratio + 1):.2f}')
        for s in STRATEGIES:
            v = R[ratio][s]
            mu = [v[[i for i, (h, _) in enumerate(keys) if h == hh]].mean()
                  for hh in horizons]
            sd = [v[[i for i, (h, _) in enumerate(keys) if h == hh]].std()
                  for hh in horizons]
            print(f'    {s:>12} | mean ' + ' '.join(f'{x:>+7.2f}' for x in mu)
                  + ' | seed-sd ' + ' '.join(f'{x:>5.2f}' for x in sd))


if __name__ == '__main__':
    main()
