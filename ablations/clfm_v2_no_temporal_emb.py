"""
CLFMv2 Ablation: No Temporal Embeddings
=======================================
Removes both time-of-day and day-of-week embeddings.
The latent field receives no explicit temporal context.

Purpose: Measure how much the temporal position encodings
         (ToD capturing rush-hour patterns, DoW capturing weekday vs weekend)
         contribute to forecasting accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..arch.mlp import MultiLayerPerceptron
from ..arch.clfm_v2 import (
    GraphLaplacianOperator,
    NeuralPDEOperator,
    ContinuousStateSpace,
    LatentFieldEncoder,
    LatentFieldDecoder,
)


class CLFMv2_NoTemporalEmb(nn.Module):
    """CLFMv2 without any temporal embeddings (ToD + DoW removed)."""

    def __init__(self, **model_args):
        super().__init__()

        self.num_nodes  = model_args["num_nodes"]
        self.input_len  = model_args["input_len"]
        self.input_dim  = model_args["input_dim"]
        self.output_len = model_args["output_len"]

        self.field_dim      = model_args.get("field_dim", 32)
        self.hidden_dim     = model_args.get("hidden_dim", 64)
        self.num_pde_steps  = model_args.get("num_pde_steps", 4)
        self.encoder_layers = model_args.get("encoder_layers", 2)
        self.decoder_layers = model_args.get("decoder_layers", 2)
        self.pde_layers     = model_args.get("pde_layers", 2)

        self.smoothness_weight = model_args.get("smoothness_weight", 0.1)

        # NO temporal embedding parameters — ABLATED
        # (if_T_i_D and if_D_i_W are ignored)

        # Encoder
        self.encoder = LatentFieldEncoder(
            input_dim=self.input_dim,
            input_len=self.input_len,
            field_dim=self.field_dim,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=self.encoder_layers,
        )

        # Static Graph Laplacian
        self.laplacian_op = GraphLaplacianOperator(num_nodes=self.num_nodes)

        # Neural PDE Operator
        self.neural_pde = NeuralPDEOperator(
            field_dim=self.field_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.pde_layers,
        )

        # Continuous State Space
        self.state_space = ContinuousStateSpace(
            field_dim=self.field_dim,
            state_dim=self.field_dim,
        )

        # Decoder
        self.decoder = LatentFieldDecoder(
            field_dim=self.field_dim,
            output_len=self.output_len,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=self.decoder_layers,
        )

        # PDE mixing weight
        self.pde_mix = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    def _pde_step(self, field, state, dt=1.0):
        alpha = torch.sigmoid(self.pde_mix)

        L_F = self.laplacian_op(field)
        N_F = self.neural_pde(field)

        dF = alpha * L_F + (1 - alpha) * N_F
        field_evolved = field + dF * dt
        field_refined, new_state = self.state_space(field_evolved, state)

        return field_refined, new_state, dF

    # ------------------------------------------------------------------
    def compute_smoothness_loss(self, derivatives):
        loss = torch.tensor(0.0, device=derivatives[0].device)
        for dF in derivatives:
            loss = loss + (dF ** 2).mean()
        return loss / max(len(derivatives), 1)

    # ------------------------------------------------------------------
    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        input_data = history_data[..., :self.input_dim]

        # Encode — NO temporal bias added
        field = self.encoder(input_data)

        # PDE evolution
        state = None
        derivatives = []
        dt = 1.0 / self.num_pde_steps

        for _ in range(self.num_pde_steps):
            field, state, dF = self._pde_step(field, state, dt)
            derivatives.append(dF)

        # Decode
        prediction = self.decoder(field)

        result = {"prediction": prediction}
        if train:
            result["smoothness_loss"] = (
                self.compute_smoothness_loss(derivatives) * self.smoothness_weight
            )
        return result
