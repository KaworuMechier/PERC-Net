#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-wide paths and constants.

Everything else derives from ``PROJECT_ROOT``, so the repository runs from
a fresh clone with no edits.  Edit ``DATA_FILE`` only if you supply your
own copy of the dataset.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Data ───────────────────────────────────────────────────────────────
# ASU campus load dataset: 23808 hourly rows, 12 raw columns
#   KW, CHWTON, HTMBTU, TEMP, DPT, SLP, SEA-P, WBT, ALT, DOY-COS,
#   CMBMBBTU, GHG
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'dataset_input.csv')

# ── Output ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ── Forecast protocol ──────────────────────────────────────────────────
SEQ_LEN = 168                      # one week of history
HORIZONS = [24, 48, 72, 96]        # four prediction horizons (hours)
TARGET_NAMES = ['Electrical Load (KW)', 'Cooling Load (CHWTON)',
                'Heating Load (HTMBTU)']
TARGET_SHORT = ['KW', 'CHW', 'HT']
SPLIT = (0.8, 0.1, 0.1)            # train / valid / test fraction
SEEDS = [2024, 2025, 2026, 2027, 2028]

# ── CEL physics identity (ASU metering units) ──────────────────────────
# CMBBTU = 0.003412*KW + 0.0120*CHWTON + 1.0000*HTMBTU  (R^2 = 1.000000)
CEL_COEF = (0.003412, 0.0120, 1.0000)
