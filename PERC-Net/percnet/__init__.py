"""PERC-Net — reference implementation package.

Submodules
----------
``percnet.model``           the paper model (frozen, self-contained)
``percnet.model_ablation``  config-driven variant behind the ablation table
``percnet.data``            sliding-window loader with NWP future covariates
``percnet.metrics``         forecast accuracy metrics (MAE/MAPE/RMSE/...)
``percnet.primitives``      shared nn.Module building blocks
``percnet.baselines``       ported comparison models (DLinear, ...)
"""

__version__ = '1.0.0'
