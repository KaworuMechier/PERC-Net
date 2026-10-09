#!/usr/bin/env bash
# ==============================================================
# PERC-Net — full reproduction suite
#
#   main table + baselines + ablations + probabilistic /
#   decision readouts + figures.
#
# Roughly 6-8 h on an RTX 3090/4090; dominated by the baseline
# sweep.  Override the sweep sizes for a quick pass, e.g.
#
#   HORIZONS=24 SEEDS=2024 bash scripts/run_all.sh
#
# Every step is resumable: run directories that already exist
# are skipped, and result CSVs are appended with dedup on the
# run key.  A single failing run does not abort the suite.
# ==============================================================
set -u
cd "$(dirname "$0")/.."          # repo root

PY="${PY:-python}"
HORIZONS="${HORIZONS:-24,48,72,96}"
SEEDS="${SEEDS:-2024,2025,2026,2027,2028}"
ABL_HORIZONS="${ABL_HORIZONS:-24}"
ABL_SEEDS="${ABL_SEEDS:-2024,2025,2026}"
BASELINES="${BASELINES:-DLinear,iTransformer,Mamba,Crossformer,FiLM,FESTRNN}"
DEVICE_ARG=()
[[ -n "${DEVICE:-}" ]] && DEVICE_ARG=(--device "$DEVICE")

step () { echo; echo "=== $(date '+%F %T') | $*"; }
run  () { "$@" || echo "  [FAIL] $* (continuing)"; }

step "0/6  install self-check"
run "$PY" scripts/smoke_test.py

step "1/6  main table — PERC-Net (nwp)"
run "$PY" -m experiments.main_table --models PERCNet --setting nwp \
    --horizons "$HORIZONS" --seeds "$SEEDS" "${DEVICE_ARG[@]}"

step "2/6  main table — baselines (base)"
run "$PY" -m experiments.main_table --models "$BASELINES" --setting base \
    --horizons "$HORIZONS" --seeds "$SEEDS" "${DEVICE_ARG[@]}"

step "3/6  ablations (all arms)"
run "$PY" -m experiments.ablation \
    --horizons "$ABL_HORIZONS" --seeds "$ABL_SEEDS" "${DEVICE_ARG[@]}"

step "4/6  multi-horizon and NWP-noise robustness"
run "$PY" -m experiments.multihorizon --seeds "$ABL_SEEDS" "${DEVICE_ARG[@]}"
run "$PY" -m experiments.nwp_noise --horizons "$HORIZONS" \
    --seeds "$ABL_SEEDS" "${DEVICE_ARG[@]}"

step "5/6  probabilistic / decision-value / identity readouts"
run "$PY" -m experiments.eval_probabilistic
run "$PY" -m experiments.eval_q50_point
run "$PY" -m experiments.eval_cel
run "$PY" -m experiments.eval_decision_value

step "6/6  efficiency and figures"
run "$PY" -m experiments.measure_efficiency "${DEVICE_ARG[@]}"
run "$PY" -m experiments.make_figures --results results --figs results/figs

echo
echo "=== $(date '+%F %T') | done"
echo "    main_results.csv          results/main_results.csv"
echo "    ablation_results.csv      results/ablation_results.csv"
echo "    probabilistic_metrics.csv results/probabilistic_metrics.csv"
echo "    efficiency_percnet.csv    results/efficiency_percnet.csv"
echo "    figures                   results/figs/"
