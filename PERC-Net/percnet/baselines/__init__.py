"""
Baselines for the Applied Energy comparison table.
====================================================
All models share the Model(config) interface:
    config: seq_len, pred_len, enc_in, c_out (+ model-specific attrs)
    forward(x (B, L, C)) → (B, H, c_out)

  DLinear       — Zeng et al., AAAI 2023 (channel-independent, ported)
  iTransformer  — Liu et al., ICLR 2024 (inverted, ported)
  Mamba         — S-Mamba style (S6 selective scan, PyTorch-scan fallback)
  Crossformer   — Zhang & Yan, ICLR 2023 (two-stage attention, compact)
  FiLM          — Zhou et al., NeurIPS 2023 (Legendre-Fourier, compact)
"""

from .dlinear import Model as DLinear
from .itransformer import Model as ITransformer
from .mamba import Model as Mamba
from .crossformer import Model as Crossformer
from .film import Model as FiLM
from .festrnn import Model as FESTRNN

MODEL_REGISTRY = {
    'DLinear': DLinear,
    'iTransformer': ITransformer,
    'Mamba': Mamba,
    'Crossformer': Crossformer,
    'FiLM': FiLM,
    'FESTRNN': FESTRNN,
}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
