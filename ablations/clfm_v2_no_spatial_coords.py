"""
CLFMv2 Ablation: No Spatial Coordinates
========================================
Removes the learnable spatial coordinate embeddings from both the
encoder and decoder.  The MLP fusion layers still exist but operate
on the projected input alone (no concatenated spatial position).

Purpose: Measure how much the per-node learnable spatial positions
         contribute.  These were the highest-norm parameters in the
         full model (encoder spatial L2 = 0.649).
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
)


# ------------------------------------------------------------------
# Modified Encoder: no spatial_coords
# ------------------------------------------------------------------
class LatentFieldEncoder_NoSpatial(nn.Module):
    """LatentFieldEncoder without learnable spatial coordinates."""

    def __init__(self, input_dim, input_len, field_dim,
                 hidden_dim, num_nodes, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim * input_len, hidden_dim)

        # NO spatial_coords — ABLATED

        # Fusion operates on hidden_dim only (not hidden_dim * 2)
        layers = []
        for i in range(num_layers):
            in_d = hidden_dim  # no concatenation
            layers.append(MultiLayerPerceptron(in_d, hidden_dim))
        self.fusion = nn.Sequential(*layers)

        self.to_field = nn.Linear(hidden_dim, field_dim)

    def forward(self, x):
        B, L, N, C = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, L * C)
        x_proj = self.input_proj(x_flat)

        # No spatial concat — feed directly
        fused = self.fusion(x_proj)

        return self.to_field(fused)


# ------------------------------------------------------------------
# Modified Decoder: no spatial_coords
# ------------------------------------------------------------------
class LatentFieldDecoder_NoSpatial(nn.Module):
    """LatentFieldDecoder without learnable spatial coordinates."""

    def __init__(self, field_dim, output_len, hidden_dim, num_nodes, num_layers=2):
        super().__init__()
        self.field_proj = nn.Linear(field_dim, hidden_dim)

        # NO spatial_coords — ABLATED

        layers = []
        for i in range(num_layers):
            in_d = hidden_dim  # no concatenation
            layers.append(MultiLayerPerceptron(in_d, hidden_dim))
        self.decoder = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, output_len)

    def forward(self, field):
        B, N, _ = field.shape
        f_proj = self.field_proj(field)

        # No spatial concat — feed directly
        decoded = self.decoder(f_proj)
        pred = self.output_proj(decoded)
        return pred.permute(0, 2, 1).unsqueeze(-1)


# ------------------------------------------------------------------
# Main ablation model
# ------------------------------------------------------------------
class CLFMv2_NoSpatialCoords(nn.Module):
    """CLFMv2 without learnable spatial coordinates in encoder/decoder."""

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

        self.if_time_in_day  = model_args.get("if_T_i_D", True)
        self.if_day_in_week  = model_args.get("if_D_i_W", True)
        self.time_of_day_size = model_args.get("time_of_day_size", 288)
        self.day_of_week_size = model_args.get("day_of_week_size", 7)
        self.temp_dim_tid    = model_args.get("temp_dim_tid", 16)
        self.temp_dim_diw    = model_args.get("temp_dim_diw", 16)

        # Temporal Embeddings (kept)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        temp_emb_dim = (self.temp_dim_tid * int(self.if_time_in_day)
                        + self.temp_dim_diw * int(self.if_day_in_week))
        if temp_emb_dim > 0:
            self.temp_to_field = nn.Linear(temp_emb_dim, self.field_dim)
        else:
            self.temp_to_field = None

        # Encoder WITHOUT spatial coords
        self.encoder = LatentFieldEncoder_NoSpatial(
            input_dim=self.input_dim,
            input_len=self.input_len,
            field_dim=self.field_dim,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=self.encoder_layers,
        )

        # Laplacian
        self.laplacian_op = GraphLaplacianOperator(num_nodes=self.num_nodes)

        # Neural PDE
        self.neural_pde = NeuralPDEOperator(
            field_dim=self.field_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.pde_layers,
        )

        # SSM
        self.state_space = ContinuousStateSpace(
            field_dim=self.field_dim,
            state_dim=self.field_dim,
        )

        # Decoder WITHOUT spatial coords
        self.decoder = LatentFieldDecoder_NoSpatial(
            field_dim=self.field_dim,
            output_len=self.output_len,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=self.decoder_layers,
        )

        self.pde_mix = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    def _get_temporal_field_bias(self, history_data):
        parts = []
        if self.if_time_in_day:
            t_i_d = history_data[..., 1]
            idx = (t_i_d[:, -1, :] * self.time_of_day_size).long().clamp(
                0, self.time_of_day_size - 1)
            parts.append(self.time_in_day_emb[idx])
        if self.if_day_in_week:
            d_i_w = history_data[..., 2]
            idx = (d_i_w[:, -1, :] * self.day_of_week_size).long().clamp(
                0, self.day_of_week_size - 1)
            parts.append(self.day_in_week_emb[idx])
        if parts and self.temp_to_field is not None:
            temp_emb = torch.cat(parts, dim=-1)
            return self.temp_to_field(temp_emb)
        return None

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

        field = self.encoder(input_data)

        temp_bias = self._get_temporal_field_bias(history_data)
        if temp_bias is not None:
            field = field + temp_bias

        state = None
        derivatives = []
        dt = 1.0 / self.num_pde_steps

        for _ in range(self.num_pde_steps):
            field, state, dF = self._pde_step(field, state, dt)
            derivatives.append(dF)

        prediction = self.decoder(field)

        result = {"prediction": prediction}
        if train:
            result["smoothness_loss"] = (
                self.compute_smoothness_loss(derivatives) * self.smoothness_weight
            )
        return result
