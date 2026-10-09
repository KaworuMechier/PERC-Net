#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training / evaluation core shared by every experiment driver.

One model, one (setting, horizon, seed) → one metrics row.  The protocol
is identical for PERC-Net and for every baseline:

* data pipeline            ``percnet.data.DataLoaderS``
* optimiser                Adam, per-model learning rate / weight decay
* schedule                 ReduceLROnPlateau on validation MAPE (recipes
                           that specify it) or cosine annealing otherwise
* model selection          best validation MAE, checkpoint restored before
                           the test pass — never test-set selection
* test metrics             computed in physical units

PERC-Net is a probabilistic model: its ``forward`` returns a dict and its
``compute_loss`` owns the objective.  For those, the main-table point
forecast is the predictive median (tau = 0.5).
"""

import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn as nn

from percnet.metrics import MAE, MAPE, RMSE, CV_RMSE, SMAPE, NMBE


# ─────────────────────────────────────────────────────────────────────────
# Device / seeding
# ─────────────────────────────────────────────────────────────────────────
def detect_device(force=None):
    """Best available device: CUDA > MPS > CPU (override with ``force``)."""
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f'  Device: CUDA — {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        dev = torch.device('mps')
        print(f'  Device: MPS ({platform.processor() or "Apple Silicon"})')
    else:
        dev = torch.device('cpu')
        print(f'  Device: CPU — {platform.processor() or "unknown"}')
    return dev


def configure_backends(device):
    """TF32 on Ampere+ (free speedup, negligible accuracy impact)."""
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def validate_mps_fft(device):
    """Fall back to CPU if this PyTorch build cannot FFT on MPS."""
    if device.type != 'mps':
        return device
    try:
        x = torch.randn(2, 168, 4, device=device)
        _ = torch.fft.rfft(x, dim=1)
        return device
    except Exception:
        print('  [WARN] MPS FFT unsupported, falling back to CPU')
        return torch.device('cpu')


def mps_warmup(model, device, seq_len, enc_in):
    """Warm the MPS compute graph — the first epoch is slow without this."""
    if device.type != 'mps':
        return
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(torch.randn(4, seq_len, enc_in, device=device))


def fix_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────
def calc_accuracy(y_true, y_pred):
    """Anomaly correlation coefficient (Pearson r on flattened arrays)."""
    yt, yp = y_true.reshape(-1), y_pred.reshape(-1)
    if yt.std() < 1e-8 or yp.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def calc_all_metrics(y_true, y_pred, n_tgt=3):
    """Total + per-target metrics.  Arrays are (N, H, C) in physical units."""
    def block(yt, yp):
        return {
            'MAE': MAE(yt, yp), 'RMSE': RMSE(yt, yp), 'MAPE': MAPE(yt, yp),
            'CV_RMSE': CV_RMSE(yt, yp), 'sMAPE': SMAPE(yt, yp),
            'NMBE': NMBE(yt, yp), 'ACCR': calc_accuracy(yt, yp),
        }

    total = block(y_true, y_pred)
    per_target = {i: block(y_true[:, :, i], y_pred[:, :, i])
                  for i in range(n_tgt)}
    return total, per_target


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, data, X, Y, batch_size, scale_arr, device, n_tgt=3,
             offset=None):
    """Batched inference → (pred, true) in physical units, shape (N, H, n_tgt)."""
    model.eval()
    preds, trues = [], []
    for Xb, Yb in data.get_batches(X, Y, batch_size, shuffle=False):
        y_true = Yb[:, :, :n_tgt].to(device)
        out = model(Xb.to(device))
        if isinstance(out, dict):          # probabilistic model
            out = out['mu']
        if out.shape[1] > y_true.shape[1]:  # model emits a longer path
            out = out[:, -y_true.shape[1]:, :]
        out = out[:, :, :y_true.shape[2]]
        preds.append(out.cpu())
        trues.append(y_true.cpu())
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    scale_t = scale_arr[:n_tgt].reshape(1, 1, n_tgt)
    off_t = (offset[:n_tgt].reshape(1, 1, n_tgt) if offset is not None else 0.0)
    return preds * scale_t + off_t, trues * scale_t + off_t


@torch.no_grad()
def collect_quantiles(model, data, batch_size, device, scale_arr,
                      n_tgt, offset_arr=None):
    """Test-set quantiles in physical units → (q (N,H,C,K), y (N,H,C), taus)."""
    qs, ys = [], []
    model.eval()
    for Xb, Yb in data.get_batches(data.test[0], data.test[1], batch_size,
                                   shuffle=False):
        qs.append(model(Xb.to(device))['quantiles'].cpu())
        ys.append(Yb.cpu())
    q_scale = scale_arr[:n_tgt].reshape(1, 1, n_tgt, 1)
    q_off = (offset_arr[:n_tgt].reshape(1, 1, n_tgt, 1)
             if offset_arr is not None else 0.0)
    y_scale = scale_arr[:n_tgt].reshape(1, 1, n_tgt)
    y_off = (offset_arr[:n_tgt].reshape(1, 1, n_tgt)
             if offset_arr is not None else 0.0)
    quantiles = torch.cat(qs).numpy() * q_scale + q_off
    y_true = torch.cat(ys).numpy()[:, :, :n_tgt] * y_scale + y_off
    taus = np.linspace(0.05, 0.95, model.K)
    return quantiles, y_true, taus


# ─────────────────────────────────────────────────────────────────────────
# One full train + evaluate pass
# ─────────────────────────────────────────────────────────────────────────
def train_and_eval(model, data, cfg, channel_w, device, scale_arr,
                   offset_arr, out_dir, batch_size, zscore=False):
    """Train to the validation-selected checkpoint, then evaluate on test.

    Returns ``(total_metrics, per_target_metrics, train_time, infer_time)``.
    Writes into ``out_dir``: ``best.pt``, ``predictions.npz`` and, for
    probabilistic models, ``quantiles.npz``.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_train, Y_train = data.train
    n_tgt = cfg.n_tgt

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weightdecay)
    if getattr(cfg, 'scheduler', 'cosine') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.plateau_factor,
            patience=cfg.plateau_patience)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 10)

    scale3 = torch.tensor(scale_arr[:n_tgt].astype(np.float32),
                          device=device).view(1, 1, n_tgt)
    best_val, best_path = float('inf'), os.path.join(out_dir, 'best.pt')
    patience = ((getattr(cfg, 'decaypatience', 8) * 3)
                if getattr(cfg, 'early_stop', True) else cfg.epochs + 1)
    loss_type = getattr(cfg, 'loss_type', 'swl1')
    clip = getattr(cfg, 'clip', 0.3)
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for Xb, Yb in data.get_batches(X_train, Y_train, batch_size,
                                       shuffle=True):
            x_enc = Xb.to(device)
            y_true = Yb[:, :, :n_tgt].to(device)
            optimizer.zero_grad()
            out = model(x_enc)
            if isinstance(out, dict):        # probabilistic model owns the loss
                loss = model.compute_loss(out, y_true, scale3)
            else:
                if out.shape[1] > y_true.shape[1]:
                    out = out[:, -y_true.shape[1]:, :]
                out = out[:, :, :y_true.shape[2]]
                if loss_type == 'mape_raw':
                    y_raw, o_raw = y_true * scale3, out * scale3
                    loss = ((y_raw - o_raw).abs()
                            / (y_raw.abs() + 1e-2)).mean() * 100
                elif loss_type == 'swl1':
                    loss = ((channel_w * (out - y_true).abs()).mean()
                            if channel_w is not None
                            else nn.functional.l1_loss(out, y_true))
                else:
                    raise ValueError(f'unknown loss_type {loss_type!r}')
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        vp, vt = evaluate(model, data, data.valid[0], data.valid[1],
                          batch_size, scale_arr, device, n_tgt=n_tgt,
                          offset=offset_arr)
        val_mae, val_mape = MAE(vt, vp), MAPE(vt, vp)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_mape)
        else:
            scheduler.step()

        if val_mae < best_val:
            best_val, no_improve = val_mae, 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f'    epoch {epoch:3d} | val MAE {val_mae:.2f} '
                  f'(best {best_val:.2f})', flush=True)
        if no_improve >= patience:
            break

    train_time = time.time() - t0
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))

    t0 = time.time()
    y_pred, y_true = evaluate(model, data, data.test[0], data.test[1],
                              batch_size, scale_arr, device, n_tgt=n_tgt,
                              offset=offset_arr)
    infer_time = time.time() - t0

    # Probabilistic models: the main-table point forecast is the predictive
    # median (tau = 0.5), the MAE-optimal elicitable point forecast.  `mu`
    # (the task-specific point output) is kept in predictions.npz so the
    # difference stays auditable.
    y_pred_q50 = None
    if getattr(cfg, 'prob_head', True) and hasattr(model, 'quantile_head'):
        quantiles, y_true_q, taus = collect_quantiles(
            model, data, batch_size, device, scale_arr, n_tgt,
            offset_arr if zscore else None)
        i50 = int(np.argmin(np.abs(taus - 0.5)))
        y_pred_q50 = quantiles[..., i50]
        np.savez(os.path.join(out_dir, 'quantiles.npz'),
                 quantiles=quantiles, y_true=y_true_q, taus=taus)

    y_point = y_pred_q50 if y_pred_q50 is not None else y_pred
    total_m, per_t = calc_all_metrics(y_true, y_point, n_tgt=n_tgt)
    np.savez(os.path.join(out_dir, 'predictions.npz'),
             y_true=y_true, y_pred=y_pred,
             **({'y_pred_q50': y_pred_q50} if y_pred_q50 is not None else {}))

    # interpretability artifacts consumed by make_figures.py
    diag = {}
    if hasattr(model, 'cycle_daily'):
        diag['cycle_daily'] = model.cycle_daily.detach().cpu().numpy()
    if hasattr(model, 'cycle_weekly'):
        diag['cycle_weekly'] = model.cycle_weekly.detach().cpu().numpy()
    if hasattr(model, 'horizon_emb'):
        diag['horizon_emb'] = model.horizon_emb.detach().cpu().numpy()
    if diag:
        np.savez(os.path.join(out_dir, 'diagnostics.npz'), **diag)

    return total_m, per_t, train_time, infer_time
