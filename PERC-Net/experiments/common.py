"""Shared experiment utilities: FLOPs counting, CSV result persistence."""

import csv
import os
import time

import torch

MAIN_HEADER = [
    'model', 'setting', 'horizon', 'seed',
    'params', 'flops_MMac', 'train_time_s', 'inference_time_s',
    'MAE', 'MAPE', 'RMSE', 'CV_RMSE', 'sMAPE', 'NMBE', 'ACCR',
    'MAE_KW', 'MAPE_KW', 'RMSE_KW', 'ACCR_KW',
    'MAE_CHW', 'MAPE_CHW', 'RMSE_CHW', 'ACCR_CHW',
    'MAE_HT', 'MAPE_HT', 'RMSE_HT', 'ACCR_HT',
]

ABLATION_HEADER = ['ablation', 'horizon', 'seed'] + MAIN_HEADER[4:]


def measure_flops(model, seq_len, enc_in, device='cpu'):
    """Matmul/conv FLOPs (MMac) for one forward via torch.profiler.
    Profiles a deepcopy on CPU — never mutates the original model's device.
    FFT/recurrent ops are not counted by the profiler — column is
    'flops_MMac' with that caveat."""
    import copy
    m = copy.deepcopy(model).to(device).eval()
    x = torch.randn(1, seq_len, enc_in, device=device)
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU],
                     with_flops=True) as prof:
            with torch.no_grad():
                m(x)
        flops = sum(e.flops for e in prof.key_averages())
        del m
        return flops / 1e6
    except Exception:
        del m
        return float('nan')


def append_csv_row(csv_path, header, row):
    """Append a dict row; write header if file is new. Returns True if
    the row was appended, False if an identical key row already exists
    (key = first 4 header fields for MAIN / first 3 for ABLATION)."""
    key_len = 4 if header is MAIN_HEADER else 3
    key = tuple(row.get(h) for h in header[:key_len])
    exists = os.path.exists(csv_path)
    if exists:
        with open(csv_path) as f:
            for r in csv.reader(f):
                if r and tuple(r[:key_len]) == tuple(str(k) for k in key):
                    return False
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([row.get(h, '') for h in header])
    return True


def build_metrics_row(model_name, setting, horizon, seed, params, flops,
                      train_time, infer_time, total_m, per_target):
    """Flatten calc_all_metrics output into a CSV row dict."""
    row = {
        'model': model_name, 'setting': setting, 'horizon': horizon,
        'seed': seed, 'params': params, 'flops_MMac': round(flops, 2),
        'train_time_s': round(train_time, 1),
        'inference_time_s': round(infer_time, 4),
    }
    for k in ('MAE', 'MAPE', 'RMSE', 'CV_RMSE', 'sMAPE', 'NMBE', 'ACCR'):
        row[k] = round(total_m[k], 4)
    tgt = {'KW': 0, 'CHW': 1, 'HT': 2}
    for short, idx in tgt.items():
        m = per_target[idx]
        for k in ('MAE', 'MAPE', 'RMSE', 'ACCR'):
            row[f'{k}_{short}'] = round(m[k], 4)
    return row


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
