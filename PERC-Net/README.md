# PERC-Net

Reference implementation and reproduction package for the paper

> **PERC-Net: A Lightweight Accounting-Consistent Probabilistic Load
> Forecaster for Integrated Energy Systems**
> Yuansheng Xu — Nanjing University of Information Science and Technology
> Submitted to *Applied Energy* (under review).

PERC-Net is a ~20 K-parameter probabilistic load-forecasting model that
composes its forecast as an **additive chain of specialists**, each
supervised against its own exclusive target, and delivers a full
predictive distribution plus an exactly-consistent system-level energy
output.

---

## Contents

| | |
|---|---|
| `percnet/model.py` | **The paper model.** Frozen, self-contained, ~470 lines. |
| `percnet/model_ablation.py` | Switchable variant behind the ablation table. |
| `percnet/data.py` | Sliding-window loader: calendar features, rolling weather statistics, NWP future covariates, forecast-error injection. |
| `percnet/metrics.py` | MAE, MAPE, RMSE, CV-RMSE, sMAPE, NMBE. |
| `percnet/baselines/` | Six comparison models (DLinear, iTransformer, Mamba, Crossformer, FiLM, FESTRNN). |
| `experiments/` | Every script that produces a number or figure in the paper. |
| `data/` | The ASU campus load dataset used throughout. |
| `scripts/smoke_test.py` | 37-check install verification (< 1 min). |

## Install

```bash
git clone https://github.com/KaworuMechier/PERC-Net.git
cd PERC-Net
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test builds the model at all four horizons, checks the
structural invariants, and — importantly — verifies that the ablation
variant is **numerically identical** to the frozen paper model under the
paper configuration. Everything is verified on CPU in under a minute.

Device selection is automatic: CUDA → Apple MPS → CPU, overridable with
`--device cpu|cuda|mps` on every driver.

## Reproducing the paper

### Main comparison table

```bash
# PERC-Net, NWP setting, 4 horizons x 5 seeds
python -m experiments.main_table --models PERCNet --setting nwp \
    --seeds 2024,2025,2026,2027,2028

# baselines, base setting
python -m experiments.main_table --models DLinear,iTransformer,Mamba,Crossformer,FiLM,FESTRNN \
    --setting base --seeds 2024,2025,2026,2027,2028
```

Rows accumulate in `results/main_results.csv`. Runs already present are
skipped, so an interrupted sweep can simply be re-launched.

### Ablations

```bash
python -m experiments.ablation --horizons 24 --seeds 2024,2025,2026
```

Every arm toggles exactly one switch away from the paper configuration:
`full`, `no_nwp`, `no_cycle`, `no_anchor`, `revin_all`, `no_nll`,
`no_cvar`, `no_component`, `no_mono`, `point_only`, `no_horizon_cond`,
`decoupled_probs`. Rows go to `results/ablation_results.csv`.

### Probabilistic and decision-level readouts

```bash
python -m experiments.eval_probabilistic       # CRPS, PICP, PINAW, Winkler, QCE, CVaR
python -m experiments.eval_q50_point           # is the median a better point forecast than mu?
python -m experiments.eval_cel                 # CEL identity: MAE and exactness
python -m experiments.eval_decision_value      # newsvendor dispatch cost of the calibrated quantiles
```

`eval_decision_value` additionally needs the point-only ablation
(`python -m experiments.ablation --ablations point_only`), because it
compares the joint configuration's quantile selection against the
point-only model's forecast.

### Robustness and multi-horizon

```bash
python -m experiments.nwp_noise     # accuracy vs. NWP forecast quality (sigma = 0 / 0.5 / 1.0)
python -m experiments.multihorizon  # ONE model serving 24/48/72/96 h
```

### Efficiency and figures

```bash
python -m experiments.measure_efficiency --device cuda
python -m experiments.make_figures --figs results/figs
```

Everything at once: `bash scripts/run_all.sh`.

---

## The model

`percnet/model.py` is a *frozen* single-configuration extract: the
ablation switches, the multi-horizon wrapper and the unused branches have
been removed, so what a reviewer reads is exactly what was run.

The forecast is composed additively, with each specialist trained against
its own target:

```
y = cycle(t) + anchor + seasonal(t) + trend(t) + local(t) + weather(t)
```

| Paper section | Where in the code |
|---|---|
| §2.2 statistical prior externalisation | `_prepare()` — RevIN, cycle queues, last-value anchor |
| §2.3 component specialist chain | `forward()` — seasonal/trend heads, patch-GRU local stream, weather branch, additive rail |
| §2.4 exclusive-target training | `compute_loss()` — the four detached component residuals |
| §2.5 probabilistic head, delivery | quantile/scale heads; delivered point forecast is the τ = 0.5 quantile |
| §2.6 lead-time-conditioned head | `horizon_emb` |
| §2.7 CEL identity output layer | the `cmbbtu` output — zero parameters, exact by construction |

Three structural invariants are asserted by `scripts/smoke_test.py`:

* the **CEL identity** `CMBBTU = 0.003412·KW + 0.0120·CHWTON + 1.0000·HTMBTU`
  holds with zero parameters and zero consistency error;
* every parameter tensor receives a gradient from `compute_loss`;
* the lead-time pathway is live (perturbing `horizon_emb` moves the
  interval-width fan ratio off 1.000).

| horizon | parameters |
|---|---|
| 24 h | 20,506 |
| 48 h | 23,152 |
| 72 h | 25,798 |
| 96 h | 28,444 |

## Input layout

`x` has shape `(B, seq_len + pred_len, 20)`: the first 168 rows are
history, the last `pred_len` rows are the future covariates supplied by
the NWP branch.

| indices | channels | normalisation |
|---|---|---|
| 0–2 | `KW`, `CHWTON`, `HTMBTU` (loads) | window-normalised (RevIN) |
| 3–9 | `TEMP DPT SLP SEA-P WBT ALT DOY-COS` | static scale |
| 10–11 | `CMBMBBTU`, `GHG` | static scale; **masked on future rows** |
| 12–15 | `sin/cos(HOD)`, `sin/cos(DOW)` | calendar, derived |
| 16–19 | `TEMP/WBT` 24 h and 72 h rolling means | derived |

Only the load channels are window-normalised: the absolute level of a
weather variable is physically informative (38 °C and 22 °C imply very
different cooling loads) and normalising it away costs accuracy.

## Dataset

`data/dataset_input.csv` — ASU campus building energy data, 23,808 hourly
rows, 12 raw columns:

```
KW, CHWTON, HTMBTU, TEMP, DPT, SLP, SEA-P, WBT, ALT, DOY-COS, CMBMBBTU, GHG
```

Split 80 / 10 / 10 by time; normalisation statistics are fit on the train
segment only. `CMBMBBTU` is an exact linear combination of the three loads
and `GHG` is largely load-driven, so **both are zeroed on the future rows**
to prevent leakage — the leakage guard is asserted in the smoke test.

To run the cross-dataset presets (`ETTm1`, `ETTh1`, `ECL`, `Exchange`),
drop the public TSLib CSVs into `tslib_datasets/`; the loader and the
model take the generic path automatically. Those datasets carry no
exogenous future covariates, so they run in the `base` setting.

## Protocol notes

* **Model selection is validation-based.** Test metrics are computed once,
  from the checkpoint with the best validation MAE. The test set is never
  touched during training or model selection.
* **Seeds.** All reported numbers are means over seeds
  `2024…2028`; the shipped `run_all.sh` uses five for the main table and
  three for the ablations.
* **The main-table point forecast of PERC-Net is the predictive median**
  (τ = 0.5), the MAE-optimal elicitable point forecast. The point output
  `mu` is also stored in `predictions.npz` so the two stay comparable —
  `eval_q50_point` quantifies the difference.
* **`nwp` vs `base`.** `nwp` feeds the `pred_len` future weather rows to
  the model as known covariates (the operational NWP setting). `base` does
  not. Baselines consume the future rows by having them appended to their
  input window.
* **Baselines run their own published recipes** (loss, optimiser,
  schedule) on our unified data pipeline — see `MODEL_RECIPES` in
  `experiments/main_table.py`.

## Repository layout

```
PERC-Net/
├── config.py                    paths, horizons, split, CEL coefficients
├── percnet/
│   ├── model.py                 ★ the paper model
│   ├── model_ablation.py        switchable variant (ablation arms)
│   ├── data.py                  sliding-window loader
│   ├── metrics.py               accuracy metrics
│   ├── primitives.py            RevIN / series decomposition
│   └── baselines/               six comparison models
├── experiments/
│   ├── harness.py               training + evaluation core
│   ├── main_table.py            main comparison table
│   ├── ablation.py              component ablations
│   ├── multihorizon.py          one model, all horizons
│   ├── nwp_noise.py             NWP forecast-error sensitivity
│   ├── eval_probabilistic.py    CRPS / PICP / PINAW / Winkler / QCE / CVaR
│   ├── eval_q50_point.py        median vs. mu as the point forecast
│   ├── eval_cel.py              CEL identity readout
│   ├── eval_decision_value.py   newsvendor dispatch cost
│   ├── measure_efficiency.py    parameters / MACs / latency
│   └── make_figures.py          paper figures
├── scripts/
│   ├── smoke_test.py            install verification (37 checks)
│   └── run_all.sh               full reproduction suite
├── data/dataset_input.csv       ASU campus load dataset
└── results/                     outputs (git-ignored)
```

## Compute

Everything runs on CPU, a single consumer GPU, or Apple Silicon. A full
run of `scripts/run_all.sh` (main table + ablations + figures) takes
roughly 6–8 hours on an RTX 3090/4090, and is dominated by the baseline
sweep. A single PERC-Net run is a few minutes.

## License

MIT — see [`LICENSE`](LICENSE).

## Contact

Yuansheng Xu — Nanjing University of Information Science and Technology
`202413170015@nuist.edu.cn` · `xys829912820@outlook.com`

Questions about the implementation or a reproduction that does not match
the paper are welcome; please open an issue.
