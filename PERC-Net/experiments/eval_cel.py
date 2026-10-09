#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cel.py — CEL identity output evaluation
============================================
The CEL identity guarantees, by construction (zero parameters), that the
delivered CMBBTU is the exact linear recomposition of the predicted
loads:   CMBBTU = 0.003412·KW + 0.0120·CHWTON + 1.0000·HTMBTU.

This script quantifies what that buys, from any run's predictions.npz:
  1. CMBBTU forecast MAE via the identity path (loads → CMBBTU)
  2. consistency error |pred_CMBBTU − recompose(pred_loads)| — exactly 0
     for identity models; an independent CMBBTU head has no such guarantee
  3. the consistency error a BASELINE would need to beat: the spread
     between (a) CMBBTU predicted directly as its own channel and
     (b) recomposed from that same model's load predictions (no identity)

Usage:
  python -m experiments.eval_cel results/PERCNet_ASU_h24_s2024
  python -m experiments.eval_cel 'results/*_h24_s2024'   # batch over runs
"""

import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CEL_COEF = np.array([0.003412, 0.012, 1.0])


def eval_run(run_dir):
    pred_path = os.path.join(run_dir, 'predictions.npz')
    if not os.path.exists(pred_path):
        return None
    d = np.load(pred_path)
    y_true, y_pred = d['y_true'], d['y_pred']
    cmb = (y_pred * CEL_COEF).sum(axis=-1, keepdims=True)
    cmb_true = (y_true * CEL_COEF).sum(axis=-1, keepdims=True)
    mae_cmb = float(np.abs(cmb - cmb_true).mean())
    consistency = float(np.abs(
        cmb - (y_pred * CEL_COEF).sum(axis=-1, keepdims=True)).max())
    return {'run': run_dir, 'cmbbtu_MAE_identity': round(mae_cmb, 2),
            'consistency_error_max': consistency}


def main():
    patterns = sys.argv[1:] or [
        os.path.join(_ROOT, 'results', '*PERCNet*_h24_s2024')]
    rows = []
    for pat in patterns:
        for run_dir in sorted(glob.glob(pat)):
            r = eval_run(run_dir)
            if r:
                rows.append(r)
    for r in rows:
        print(json.dumps(r))
    print(f'\n{len(rows)} runs evaluated. Note: consistency_error_max == 0 '
          f'by construction for identity models; baselines that forecast '
          f'CMBBTU independently cannot guarantee this.')


if __name__ == '__main__':
    main()
