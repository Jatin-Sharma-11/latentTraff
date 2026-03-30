"""
CLFMv2 Ablation Study
=====================
Each variant removes exactly ONE component from the full CLFMv2 model.

Variants:
    CLFMv2_NoLaplacian      — Remove graph Laplacian diffusion (α·L·F term)
    CLFMv2_NoNeuralPDE      — Remove neural PDE operator ((1-α)·N_θ(F) term)
    CLFMv2_NoSSM            — Remove continuous state-space temporal refinement
    CLFMv2_NoTemporalEmb    — Remove time-of-day & day-of-week embeddings
    CLFMv2_NoSpatialCoords  — Remove learnable spatial coordinates in encoder/decoder
    CLFMv2_NoSmoothnessLoss — Remove PDE smoothness regularisation
"""

from .clfm_v2_no_laplacian import CLFMv2_NoLaplacian
from .clfm_v2_no_neural_pde import CLFMv2_NoNeuralPDE
from .clfm_v2_no_ssm import CLFMv2_NoSSM
from .clfm_v2_no_temporal_emb import CLFMv2_NoTemporalEmb
from .clfm_v2_no_spatial_coords import CLFMv2_NoSpatialCoords
from .clfm_v2_no_smoothness_loss import CLFMv2_NoSmoothnessLoss

__all__ = [
    "CLFMv2_NoLaplacian",
    "CLFMv2_NoNeuralPDE",
    "CLFMv2_NoSSM",
    "CLFMv2_NoTemporalEmb",
    "CLFMv2_NoSpatialCoords",
    "CLFMv2_NoSmoothnessLoss",
]
