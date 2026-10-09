import math
import numpy as np


def MAE(y_true, y_pre):
    y_true = y_true.reshape((-1, 1))
    y_pre = y_pre.reshape((-1, 1))
    return float(np.abs(y_true - y_pre).mean())


def RMSE(y_true, y_pre):
    y_true = y_true.reshape((-1, 1))
    y_pre = y_pre.reshape((-1, 1))
    return float(math.sqrt(((y_true - y_pre) ** 2).mean()))


def MAPE(y_true, y_pre, zero_thr=1e-3):
    """Masked MAPE (%): elements with |y_true| < zero_thr are excluded —
    near-zero targets (nighttime lows, ETT LUFL/LULL) make plain MAPE
    explode; the energy-forecasting community masks such periods."""
    y_true = y_true.reshape((-1, 1))
    y_pre = y_pre.reshape((-1, 1))
    mask = np.abs(y_true) > zero_thr
    if mask.sum() == 0:
        return 0.0
    re = np.mean(np.abs((y_true[mask] - y_pre[mask]) / y_true[mask])) * 100
    return float(re)


def CV_RMSE(y_true, y_pre):
    """Coefficient of variation of RMSE (ASHRAE Guideline 14, %)."""
    y_true = y_true.reshape((-1, 1))
    y_pre = y_pre.reshape((-1, 1))
    rmse = math.sqrt(((y_true - y_pre) ** 2).mean())
    mean = np.abs(y_true).mean()
    return float(rmse / (mean + 1e-8) * 100)


def NMBE(y_true, y_pre):
    """Normalized mean bias error (ASHRAE Guideline 14, %)."""
    y_true = y_true.reshape((-1, 1))
    y_pre = y_pre.reshape((-1, 1))
    mean = np.abs(y_true).mean()
    return float((y_true - y_pre).sum() / (len(y_true) * mean + 1e-8) * 100)


def SMAPE(y_true, y_pre):
    """Symmetric MAPE (%), robust to near-zero true values."""
    y_true = y_true.reshape((-1, 1))
    y_pre = y_pre.reshape((-1, 1))
    denom = (np.abs(y_true) + np.abs(y_pre)) / 2.0
    return float(np.mean(np.abs(y_true - y_pre) / (denom + 1e-8)) * 100)
