"""
CLFMv2 Ablation: No Smoothness Loss
====================================
Removes the PDE smoothness regularisation  L_smooth = (1/K) Σ_k ‖dF_k‖².
The model trains with MAE only — no penalty on erratic field evolution.

Purpose: Measure whether the smoothness prior on field derivatives
         actually helps regularise training and improves generalisation,
         or if the model learns smooth dynamics on its own.
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


class CLFMv2_NoSmoothnessLoss(nn.Module):
    """CLFMv2 without PDE smoothness loss (train with MAE only)."""

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

        # NO smoothness_weight — ABLATED

        self.if_time_in_day  = model_args.get("if_T_i_D", True)
        self.if_day_in_week  = model_args.get("if_D_i_W", True)
        self.time_of_day_size = model_args.get("time_of_day_size", 288)
        self.day_of_week_size = model_args.get("day_of_week_size", 7)
        self.temp_dim_tid    = model_args.get("temp_dim_tid", 16)
        self.temp_dim_diw    = model_args.get("temp_dim_diw", 16)

        # Temporal Embeddings
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

        # Encoder
        self.encoder = LatentFieldEncoder(
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

        # Decoder
        self.decoder = LatentFieldDecoder(
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
    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        input_data = history_data[..., :self.input_dim]

        field = self.encoder(input_data)

        temp_bias = self._get_temporal_field_bias(history_data)
        if temp_bias is not None:
            field = field + temp_bias

        state = None
        dt = 1.0 / self.num_pde_steps

        for _ in range(self.num_pde_steps):
            field, state, _ = self._pde_step(field, state, dt)

        prediction = self.decoder(field)

        # NO smoothness_loss returned — train with MAE only
        result = {"prediction": prediction}
        return result
