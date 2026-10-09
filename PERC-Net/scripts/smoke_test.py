#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Installation self-check.

Verifies, in order:

1. the data loader produces the documented input layout
   (168 history rows, 168 + H rows in the NWP setting, 20 channels);
2. the frozen paper model builds at every horizon and every parameter
   tensor receives a gradient from ``compute_loss``;
3. the exact CEL identity holds (zero parameters, zero consistency error);
4. the config-driven ablation model, given the paper configuration, is
   *numerically identical* to the frozen paper model — same parameter
   count and same forward output under a transferred state dict.  This is
   what makes the ablation table's ``full`` arm a valid reference;
5. one short training step runs end to end through the shared harness.

Exit status is non-zero if any check fails.

    python scripts/smoke_test.py
"""

import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_FILE, HORIZONS, SEQ_LEN
from percnet.data import DataLoaderS
from percnet import model as paper_model
from percnet import model_ablation as abl_model
from experiments.main_table import build_config, instantiate

ENC_IN = 20
ALLOW_CPU = True

_checks = []


def check(name, ok, detail=''):
    _checks.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))


def paper_kwargs(h):
    """The keyword arguments that define the paper model at horizon ``h``."""
    return dict(seq_len=SEQ_LEN, pred_len=h, enc_in=ENC_IN, n_free=3,
                hidden=36, decomp_kernel=13, seg_len=24, d_model=12,
                dropout=0.1, n_quantiles=9, beta=0.9, lam_pin=1.0,
                lam_nll=1.0, lam_tail=0.5, lam_mono=0.1, lam_dec=10.0)


def main():
    torch.manual_seed(0)
    device = torch.device('cpu')

    print('\n1. data pipeline')
    check('dataset present', os.path.exists(DATA_FILE), DATA_FILE)
    base = DataLoaderS(DATA_FILE, 0.8, 0.1, device, 24, SEQ_LEN, normalize=2,
                       multi_season_val=False, use_calendar=True,
                       use_future_weather=False, use_rolling_weather=True)
    nwp = DataLoaderS(DATA_FILE, 0.8, 0.1, device, 24, SEQ_LEN, normalize=2,
                      multi_season_val=False, use_calendar=True,
                      use_future_weather=True, use_rolling_weather=True)
    check('base window shape', tuple(base.train[0].shape[1:]) == (SEQ_LEN, ENC_IN),
          f'{tuple(base.train[0].shape)}')
    check('nwp window shape',
          tuple(nwp.train[0].shape[1:]) == (SEQ_LEN + 24, ENC_IN),
          f'{tuple(nwp.train[0].shape)}')
    # leakage guard: the future rows must zero the load and derived columns
    fut = nwp.train[0][:, SEQ_LEN:, :]
    check('future load columns masked', float(fut[:, :, :3].abs().max()) == 0.0)
    check('future CEL/GHG columns masked',
          float(fut[:, :, 10:12].abs().max()) == 0.0)
    check('train/valid/test non-empty',
          min(len(base.train[0]), len(base.valid[0]), len(base.test[0])) > 0,
          f'{len(base.train[0])}/{len(base.valid[0])}/{len(base.test[0])}')

    print('\n2. frozen paper model — build, invariants, gradients')
    for h in HORIZONS:
        m = paper_model.Model(**paper_kwargs(h)).eval()
        n_param = sum(p.numel() for p in m.parameters())
        x = torch.randn(4, SEQ_LEN + h, ENC_IN)
        with torch.no_grad():
            out = m(x)
        mu, q, sig, cm = (out['mu'], out['quantiles'], out['sigma'],
                          out['cmbbtu'])
        shapes_ok = (mu.shape == (4, h, 3) and q.shape == (4, h, 3, 9)
                     and sig.shape == (4, h, 3) and cm.shape == (4, h, 1))

        cm_ref = (mu * m.load_scale.view(1, 1, -1)
                  * (m.cel_coef * m.load_scale).view(1, 1, -1)
                  ).sum(-1, keepdim=True)
        cel_err = float((cm - cm_ref).abs().max())

        m.train()
        y = torch.randn(4, h, 3)
        loss = m.compute_loss(m(x), y, torch.ones(3))
        loss.backward()
        params = list(m.parameters())
        n_grad = sum(p.grad is not None for p in params)

        check(f'h={h} params / shapes', shapes_ok,
              f'{n_param:,} params, mu{tuple(mu.shape)} q{tuple(q.shape)}')
        check(f'h={h} CEL identity exact', cel_err == 0.0, f'max err {cel_err:.1e}')
        check(f'h={h} all parameters receive gradient',
              n_grad == len(params), f'{n_grad}/{len(params)}')
        check(f'h={h} loss finite', np.isfinite(loss.item()),
              f'loss {loss.item():.2f}')

    print('\n3. ablation model == paper model under the paper configuration')
    for h in HORIZONS:
        fin = paper_model.Model(**paper_kwargs(h)).eval()
        cfg = build_config('PERCNetAbl', h, 'nwp')
        abl = abl_model.Model(cfg).to(device).eval()
        n_fin = sum(p.numel() for p in fin.parameters())
        n_abl = sum(p.numel() for p in abl.parameters())
        same_params = (n_fin == n_abl)
        try:
            abl.load_state_dict(fin.state_dict())
            loaded = True
        except RuntimeError as e:
            loaded = False
            detail = str(e).splitlines()[0]
        x = torch.randn(4, SEQ_LEN + h, ENC_IN)
        if loaded:
            with torch.no_grad():
                a, b = fin(x)['mu'], abl(x)['mu']
            d = float((a - b).abs().max())
            rel = d / max(float(a.abs().max()), 1e-8)
        else:
            d, rel = float('inf'), float('inf')
        check(f'h={h} parameter count matches', same_params,
              f'{n_fin:,} vs {n_abl:,}')
        check(f'h={h} state dict transfers', loaded if not loaded else True)
        check(f'h={h} forward output identical', rel < 1e-6,
              f'max|dmu| = {d:.2e} (rel {rel:.1e})')

    print('\n4. end-to-end training step through the shared harness')
    try:
        import tempfile
        from experiments.harness import train_and_eval
        cfg = build_config('PERCNet', 24, 'nwp')
        cfg.epochs = 1
        cfg.batchsize = 16
        scale_arr = nwp.scale.cpu().numpy()
        offset_arr = getattr(nwp, 'offset', np.zeros_like(scale_arr))
        m, _ = instantiate('PERCNet', cfg, device, scale_arr, 'ASU')
        # artifacts go to a throwaway directory, so the smoke test never
        # leaves a bogus "run" behind for the eval scripts to pick up
        out_dir = tempfile.mkdtemp(prefix='percnet_smoketest_')
        total, per_t, tt, it = train_and_eval(
            m, nwp, cfg, None, device, scale_arr, offset_arr, out_dir,
            cfg.batchsize)
        check('train_and_eval completes', np.isfinite(total['MAE']),
              f"1-epoch val-selected MAE {total['MAE']:.2f}")
        check('predictions.npz written',
              os.path.exists(os.path.join(out_dir, 'predictions.npz')))
        check('quantiles.npz written',
              os.path.exists(os.path.join(out_dir, 'quantiles.npz')))
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception as e:
        check('train_and_eval completes', False, f'{type(e).__name__}: {e}')

    failed = [n for n, ok in _checks if not ok]
    print(f'\n{len(_checks) - len(failed)}/{len(_checks)} checks passed')
    if failed:
        print('FAILED: ' + ', '.join(failed))
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
