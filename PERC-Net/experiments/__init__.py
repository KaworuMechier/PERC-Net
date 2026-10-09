"""Reproduction drivers for the PERC-Net paper.

``harness``      device/seed helpers, evaluation, one training+eval pass
``main_table``   main comparison table (PERC-Net + baselines)
``ablation``     component ablations, all run against the paper model
``multihorizon`` one model serving 24/48/72/96 h
``nwp_noise``    NWP forecast-error sensitivity sweep
``eval_*``       probabilistic / decision-value / physics-identity readouts
``measure_efficiency``  params / MACs / latency
``make_figures``        paper figures from saved run artifacts
"""
