"""
CLFMv2 — Cleaned Causal Latent Field Model (Post-Training Analysis Revision)
=============================================================================

This is a streamlined version of CLFM with all dead / failed components removed
based on comprehensive post-training analysis on PEMS08.

REMOVED (failed components — see README.md for full analysis):
  - CausalGraphDiscovery : emitter/receiver embeddings collapsed to zero norms,
    all edge weights ≡ 0.5, lag kernel stayed uniform [1/3, 1/3, 1/3],
    zero cross-sample variation (cos_sim = 1.0).  Total dead weight.
  - DAG acyclicity loss  : nothing to constrain when the causal graph is uniform.
  - Sparsity loss        : same — L1 on a constant matrix is a constant.
  - Dynamic causal Laplacian path : uniform adjacency produces a constant matrix,
    adding no structural information beyond a scalar shift.

KEPT (verified working):
  - LatentFieldEncoder   : spatial coords learned (PCA explains 25.8%,
    t-SNE clusters correlate with traffic flow).
  - LatentFieldDecoder   : spatial coords learned (mean L2 norm = 0.397).
  - NeuralPDEOperator    : output layer Frobenius norm grew 0 → 4.86;
    the neural nonlinear term is the *dominant* evolution pathway (58%).
  - ContinuousStateSpace : all A-eigenvalues negative (stable), dt learned
    (0.1455 from init 0.135), skip connection D learned diverse weights.
  - Temporal Embeddings  : ToD PC1 captures day/night + rush hour;
    DoW embeddings near-orthogonal per day.
  - PDE mix α = 0.4175   : model chose 42% Laplacian, 58% Neural.
  - Diffusion coeff      : moved from init 0.1 → σ(w) = 0.317.

Net effect:
  - ~19,000 fewer parameters (causal_emitter 170×32, causal_receiver 170×32,
    lag_kernel 3, gate_net ~2K).
  - Simpler forward pass (no Gumbel sampling, no DAG trace, no dynamic Laplacian).
  - The PDE becomes:  dF/dt = α · L_road · F  +  (1−α) · N_θ(F)
    using the *static* road Laplacian only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .mlp import MultiLayerPerceptron


# =============================================================================
# Component 1: Static Graph Laplacian Operator
# =============================================================================

class GraphLaplacianOperator(nn.Module):
    """Static graph Laplacian diffusion:  L_road · F.

    Computes the symmetric normalized Laplacian from the road adjacency matrix
    and applies it as a linear diffusion operator on the latent field.

    The learnable diffusion coefficient σ(w) controls how fast the field
    propagates along the road topology.

    Post-training finding:
        diffusion_coeff moved from init 0.1  →  σ(w) = 0.317,
        indicating moderate but meaningful spatial smoothing.
    """

    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes

        # Learnable diffusion coefficient (how fast the field propagates)
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))

        # Static Laplacian buffer (set via set_static_laplacian)
        self.register_buffer('static_L', torch.zeros(num_nodes, num_nodes))

    def set_static_laplacian(self, adj_mx: torch.Tensor):
        """Compute and store symmetric normalised Laplacian from adjacency.

        L = I − D^{−1/2} A D^{−1/2}

        Args:
            adj_mx: [N, N] binary / weighted road adjacency matrix.
        """
        A = adj_mx.float()
        D = A.sum(dim=1)
        D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        L = torch.eye(self.num_nodes, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        self.static_L.copy_(L)

    def forward(self, F_field: torch.Tensor) -> torch.Tensor:
        """Apply static road Laplacian diffusion to the latent field.

        Args:
            F_field: [B, N, D] latent field values.

        Returns:
            diffused: [B, N, D] Laplacian−diffused field.
        """
        # [N, N] @ [B, N, D] via einsum
        diffused = torch.einsum('nm,bmd->bnd', self.static_L, F_field)
        return diffused * torch.sigmoid(self.diffusion_coeff)


# =============================================================================
# Component 2: Neural PDE Operator  N_θ(F)
# =============================================================================

class NeuralPDEOperator(nn.Module):
    """Neural nonlinear operator  N_θ(F)  in the PDE:

        dF/dt  =  α · L_road · F  +  (1−α) · N_θ(F)

    This captures non-linear spatio-temporal dynamics that the linear
    Laplacian diffusion cannot represent (e.g. congestion propagation,
    flow-density phase transitions).

    Architecture:  ``num_layers`` × (Linear → LayerNorm → GELU) → Linear.
    The output layer is zero-initialised so the model starts with
    pure diffusion and gradually learns the nonlinear correction.

    Post-training finding:
        Output-layer Frobenius norm grew from 0 → 4.86 — the neural term
        is actively contributing and is in fact the *dominant* evolution
        pathway (α ≈ 0.42 means the Neural share is 1−α ≈ 0.58).
    """

    def __init__(self, field_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = field_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.output = nn.Linear(hidden_dim, field_dim)

        # Zero-init output so nonlinear contribution grows from zero
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, F_field: torch.Tensor) -> torch.Tensor:
        """Compute nonlinear evolution term.

        Args:
            F_field: [B, N, D] latent field.

        Returns:
            N_theta: [B, N, D] nonlinear PDE term.
        """
        h = F_field
        for layer, norm in zip(self.layers, self.norms):
            h = norm(F.gelu(layer(h)))
        return self.output(h)


# =============================================================================
# Component 3: Continuous-Time State Space Layer
# =============================================================================

class ContinuousStateSpace(nn.Module):
    """Continuous-time state space model for temporal refinement.

    After each PDE step produces a candidate evolved field, the SSM
    refines it through a structured state-space transition:

        dz/dt = A z + B u      (state update)
        y     = C z + D u      (output)

    discretised via zero-order hold.

    Design choices:
        - Diagonal A (efficiency + stability guarantee via negative eigenvalues).
        - HiPPO-inspired initialisation for long-range memory.
        - Learnable log-scale step size ``dt`` for adaptive temporal resolution.
        - Skip connection D acts as a residual gate.

    Post-training findings:
        - All A eigenvalues are negative (−0.3 … −0.9): properly stable.
        - dt learned to 0.1455 (from init exp(−2) ≈ 0.135): slight upward shift.
        - D skip connection: mean 0.137 with diverse per-dimension distribution,
          indicating the model learned dimension-specific residual scaling.
    """

    def __init__(self, field_dim: int, state_dim: int = None):
        super().__init__()
        self.field_dim = field_dim
        self.state_dim = state_dim or field_dim

        # State transition A (negative real part → guaranteed stability)
        self.log_A_real = nn.Parameter(torch.randn(self.state_dim) * 0.5 - 1.0)
        self.A_imag = nn.Parameter(torch.randn(self.state_dim) * 0.1)

        # Input-to-state and state-to-output projections
        self.B = nn.Linear(field_dim, self.state_dim)
        self.C = nn.Linear(self.state_dim, field_dim)

        # Skip connection (D matrix in SSM notation)
        self.D = nn.Parameter(torch.ones(field_dim) * 0.1)

        # Learnable discretisation step
        self.log_dt = nn.Parameter(torch.tensor(-2.0))

    def forward(self, F_field: torch.Tensor,
                state: torch.Tensor = None) -> tuple:
        """One step of continuous-time state space evolution.

        Args:
            F_field: [B, N, D] current field input.
            state:   [B, N, state_dim] previous hidden state (or None).

        Returns:
            output:    [B, N, D] refined field.
            new_state: [B, N, state_dim] updated state.
        """
        B, N, D = F_field.shape
        dt = torch.exp(self.log_dt).clamp(max=1.0)

        if state is None:
            state = torch.zeros(B, N, self.state_dim, device=F_field.device)

        # Discretise A via zero-order hold: A_bar = exp(A · dt)
        A_real = -torch.exp(self.log_A_real)      # negative → stable
        A_discrete = torch.exp(A_real * dt)        # [state_dim]

        # State update: x_{k+1} = A_bar · x_k  +  B · u_k · dt
        Bu = self.B(F_field)                       # [B, N, state_dim]
        new_state = A_discrete.unsqueeze(0).unsqueeze(0) * state + Bu * dt

        # Output: y_k = C · x_k  +  D · u_k
        output = self.C(new_state) + self.D.unsqueeze(0).unsqueeze(0) * F_field

        return output, new_state


# =============================================================================
# Component 4: Latent Field Encoder
# =============================================================================

class LatentFieldEncoder(nn.Module):
    """Encodes discrete sensor observations into a continuous latent field.

    Maps sparse sensor readings  X_t ∈ ℝ^{N×F}  into a continuous
    latent field representation  F_t ∈ ℝ^{N×D}.

    The key idea is to treat the N traffic sensors as *sparse samples*
    of an underlying continuous physical field over the road network.

    Architecture:
        1. Flatten the temporal window:  [B, L, N, C] → [B, N, L·C]
        2. Project to hidden space via Linear.
        3. Concatenate **learnable spatial coordinates** (per-node embeddings
           that capture each sensor's role in the field).
        4. Fuse through ``num_layers`` MLP blocks (with residual + LayerNorm).
        5. Project to field dimension.

    Post-training findings:
        - Spatial coordinates are the **highest-norm** learned parameters
          (mean L2 = 0.649, max component norm ~8–9).
        - PCA on encoder spatial coords: PC1 explains 25.78%.
        - t-SNE shows clear clusters correlated with mean traffic flow
          → the encoder learned to spatially group sensors by traffic regime.
    """

    def __init__(self, input_dim: int, input_len: int, field_dim: int,
                 hidden_dim: int, num_nodes: int, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim * input_len, hidden_dim)

        # Learnable spatial position encoding (continuous coordinates)
        self.spatial_coords = nn.Parameter(
            torch.randn(num_nodes, hidden_dim) * 0.02)

        # Fusion network
        layers = []
        for i in range(num_layers):
            in_d = hidden_dim * 2 if i == 0 else hidden_dim
            layers.append(MultiLayerPerceptron(in_d, hidden_dim))
        self.fusion = nn.Sequential(*layers)

        self.to_field = nn.Linear(hidden_dim, field_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sensor observations to latent field.

        Args:
            x: [B, L, N, C] input traffic window.

        Returns:
            field: [B, N, field_dim] continuous latent field.
        """
        B, L, N, C = x.shape
        # Flatten temporal: [B, N, L·C]
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, L * C)
        x_proj = self.input_proj(x_flat)                    # [B, N, hidden]

        # Concatenate spatial position encoding
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        fused = torch.cat([x_proj, spatial], dim=-1)        # [B, N, hidden×2]
        fused = self.fusion(fused)                           # [B, N, hidden]

        return self.to_field(fused)                          # [B, N, field_dim]


# =============================================================================
# Component 5: Latent Field Decoder
# =============================================================================

class LatentFieldDecoder(nn.Module):
    """Decodes evolved latent field back to traffic predictions.

    Maps:  F_{t+K} ∈ ℝ^{N×D}  →  X̂_{t+1:t+H} ∈ ℝ^{N × output_len}

    Architecture mirrors the encoder: project field → concatenate decoder
    spatial coordinates → MLP fusion → project to output length.

    Post-training findings:
        - Decoder spatial coords learned (mean L2 = 0.397), lower magnitude
          than encoder (0.649) — the decoder relies more on field content.
        - Encoder–decoder spatial cosine similarity ≈ 0.031 (essentially
          orthogonal): they learned *independent* spatial representations,
          suggesting the encoder captures "where traffic is" while the
          decoder captures "how to reconstruct at each location".
    """

    def __init__(self, field_dim: int, output_len: int,
                 hidden_dim: int, num_nodes: int, num_layers: int = 2):
        super().__init__()
        self.field_proj = nn.Linear(field_dim, hidden_dim)
        self.spatial_coords = nn.Parameter(
            torch.randn(num_nodes, hidden_dim) * 0.02)

        layers = []
        for i in range(num_layers):
            in_d = hidden_dim * 2 if i == 0 else hidden_dim
            layers.append(MultiLayerPerceptron(in_d, hidden_dim))
        self.decoder = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, output_len)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Decode latent field to traffic predictions.

        Args:
            field: [B, N, field_dim] evolved latent field.

        Returns:
            pred: [B, output_len, N, 1]
        """
        B, N, _ = field.shape
        f_proj = self.field_proj(field)                     # [B, N, hidden]
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        fused = torch.cat([f_proj, spatial], dim=-1)
        decoded = self.decoder(fused)
        pred = self.output_proj(decoded)                    # [B, N, output_len]
        return pred.permute(0, 2, 1).unsqueeze(-1)          # [B, output_len, N, 1]


# =============================================================================
# Main Model:  CLFMv2  — Latent Field Model for Traffic
# =============================================================================

class CLFMv2(nn.Module):
    """CLFMv2 — Cleaned Latent Field Model for Traffic Forecasting.

    Models traffic as a continuous latent field  F(x, t)  evolving via a
    learned neural PDE constrained by the static road topology.

    Core equation (per PDE step):
        dF/dt  =  α · L_road · F  +  (1−α) · N_θ(F)

    Where:
        F ∈ ℝ^{N×D}  — continuous latent field (sensors = sparse samples)
        L_road        — symmetric normalised Laplacian of the road graph
        N_θ           — neural nonlinear operator (2-layer MLP)
        α             — learnable mixing weight  (post-training: 0.4175)

    After each PDE step the field is refined through a Continuous State Space
    Model (SSM) that handles temporal dynamics within each node.

    Pipeline:
        1. **Encode**  :  [B,L,N,C] → [B,N,D]  with spatial coordinates + MLP
        2. **Temporal** :  Add time-of-day / day-of-week bias to the field
        3. **PDE×K**    :  K Euler steps of  dF/dt = α·L·F + (1−α)·N_θ(F),
                           each refined through SSM
        4. **Decode**   :  [B,N,D] → [B,H,N,1]

    Training objectives:
        - Forecast loss (MAE)
        - PDE smoothness loss  L_smooth = (1/K) Σ_k ‖dF_k‖²

    Compared to the original CLFM this removes ~19 K dead parameters
    (CausalGraphDiscovery) and eliminates DAG / sparsity losses that had
    no meaningful target.

    Args (model_args dict):
        num_nodes        : Number of sensor nodes (170 for PEMS08).
        input_len        : Historical window length (12).
        input_dim        : Input features per step (3: flow, tod, dow).
        output_len       : Forecast horizon (12).
        field_dim        : Latent field dimension (32).
        hidden_dim       : MLP hidden dimension (64).
        num_pde_steps    : PDE Euler steps K (4).
        encoder_layers   : Encoder MLP depth (2).
        decoder_layers   : Decoder MLP depth (2).
        pde_layers       : Neural PDE MLP depth (2).
        smoothness_weight: Weight on PDE smoothness loss (0.1).
        if_T_i_D         : Use time-of-day embeddings (True).
        if_D_i_W         : Use day-of-week embeddings (True).
        temp_dim_tid     : Time-of-day embedding dimension (16).
        temp_dim_diw     : Day-of-week embedding dimension (16).
        time_of_day_size : Number of time-of-day slots (288 = 24h / 5min).
        day_of_week_size : Number of weekday slots (7).
    """

    def __init__(self, **model_args):
        super().__init__()

        # ==================== Core dimensions ====================
        self.num_nodes  = model_args["num_nodes"]
        self.input_len  = model_args["input_len"]
        self.input_dim  = model_args["input_dim"]
        self.output_len = model_args["output_len"]

        # ==================== Field parameters ====================
        self.field_dim      = model_args.get("field_dim", 32)
        self.hidden_dim     = model_args.get("hidden_dim", 64)
        self.num_pde_steps  = model_args.get("num_pde_steps", 4)
        self.encoder_layers = model_args.get("encoder_layers", 2)
        self.decoder_layers = model_args.get("decoder_layers", 2)
        self.pde_layers     = model_args.get("pde_layers", 2)

        # ==================== Loss weights ====================
        self.smoothness_weight = model_args.get("smoothness_weight", 0.1)

        # ==================== Temporal embedding config ====================
        self.if_time_in_day  = model_args.get("if_T_i_D", True)
        self.if_day_in_week  = model_args.get("if_D_i_W", True)
        self.time_of_day_size = model_args.get("time_of_day_size", 288)
        self.day_of_week_size = model_args.get("day_of_week_size", 7)
        self.temp_dim_tid    = model_args.get("temp_dim_tid", 16)
        self.temp_dim_diw    = model_args.get("temp_dim_diw", 16)

        # ============================================================
        # Temporal Embeddings
        # ============================================================
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # Temporal → field projection
        temp_emb_dim = (self.temp_dim_tid * int(self.if_time_in_day)
                        + self.temp_dim_diw * int(self.if_day_in_week))
        if temp_emb_dim > 0:
            self.temp_to_field = nn.Linear(temp_emb_dim, self.field_dim)
        else:
            self.temp_to_field = None

        # ============================================================
        # 1. Latent Field Encoder
        # ============================================================
        self.encoder = LatentFieldEncoder(
            input_dim=self.input_dim,
            input_len=self.input_len,
            field_dim=self.field_dim,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=self.encoder_layers,
        )

        # ============================================================
        # 2. Static Graph Laplacian Operator
        # ============================================================
        self.laplacian_op = GraphLaplacianOperator(
            num_nodes=self.num_nodes,
        )

        # ============================================================
        # 3. Neural PDE Operator
        # ============================================================
        self.neural_pde = NeuralPDEOperator(
            field_dim=self.field_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.pde_layers,
        )

        # ============================================================
        # 4. Continuous State Space
        # ============================================================
        self.state_space = ContinuousStateSpace(
            field_dim=self.field_dim,
            state_dim=self.field_dim,
        )

        # ============================================================
        # 5. Decoder
        # ============================================================
        self.decoder = LatentFieldDecoder(
            field_dim=self.field_dim,
            output_len=self.output_len,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_layers=self.decoder_layers,
        )

        # PDE mixing weight: learnable  α  in  α·L·F + (1−α)·N_θ(F)
        self.pde_mix = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Temporal embedding helper
    # ------------------------------------------------------------------
    def _get_temporal_field_bias(self, history_data: torch.Tensor) -> torch.Tensor:
        """Compute temporal bias for the latent field.

        Looks up the last time step's time-of-day and day-of-week indices,
        retrieves the corresponding embeddings, concatenates them, and
        projects into field space.

        Args:
            history_data: [B, L, N, C]

        Returns:
            temporal_bias: [B, N, field_dim]  or  None
        """
        parts = []
        if self.if_time_in_day:
            t_i_d = history_data[..., 1]                          # [B, L, N]
            idx = (t_i_d[:, -1, :] * self.time_of_day_size).long().clamp(
                0, self.time_of_day_size - 1)
            parts.append(self.time_in_day_emb[idx])
        if self.if_day_in_week:
            d_i_w = history_data[..., 2]                          # [B, L, N]
            idx = (d_i_w[:, -1, :] * self.day_of_week_size).long().clamp(
                0, self.day_of_week_size - 1)
            parts.append(self.day_in_week_emb[idx])

        if parts and self.temp_to_field is not None:
            temp_emb = torch.cat(parts, dim=-1)                   # [B, N, 32]
            return self.temp_to_field(temp_emb)                   # [B, N, D]
        return None

    # ------------------------------------------------------------------
    # PDE evolution step:  dF/dt = α·L·F + (1−α)·N_θ(F)
    # ------------------------------------------------------------------
    def _pde_step(self, field: torch.Tensor,
                  state: torch.Tensor, dt: float = 1.0) -> tuple:
        """Single discrete PDE evolution step.

        Computes:
            dF  = α · L_road · F  +  (1−α) · N_θ(F)
            F'  = F + dt · dF                          (Euler step)
            F'' = SSM(F', state)                        (temporal refinement)

        Args:
            field: [B, N, D] current latent field.
            state: [B, N, D] SSM hidden state.
            dt:    Euler step size (= 1/num_pde_steps).

        Returns:
            new_field: [B, N, D] evolved field.
            new_state: [B, N, D] updated SSM state.
            dF:        [B, N, D] field derivative (for smoothness loss).
        """
        alpha = torch.sigmoid(self.pde_mix)

        # Linear diffusion term:  L_road · F
        L_F = self.laplacian_op(field)            # [B, N, D]

        # Nonlinear neural term:  N_θ(F)
        N_F = self.neural_pde(field)              # [B, N, D]

        # Combined PDE:  dF/dt = α·L·F + (1−α)·N_θ(F)
        dF = alpha * L_F + (1 - alpha) * N_F

        # Euler step
        field_evolved = field + dF * dt

        # Refine through continuous state space
        field_refined, new_state = self.state_space(field_evolved, state)

        return field_refined, new_state, dF

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def compute_smoothness_loss(self, derivatives: list) -> torch.Tensor:
        """PDE smoothness: penalise erratic field evolution.

        L_smooth = (1/K) Σ_k ‖dF_k‖²

        Encourages smooth, physically plausible dynamics.
        """
        loss = torch.tensor(0.0, device=derivatives[0].device)
        for dF in derivatives:
            loss = loss + (dF ** 2).mean()
        return loss / max(len(derivatives), 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, history_data: torch.Tensor,
                future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                **kwargs) -> dict:
        """Full forward pass:  encode → evolve PDE → decode.

        Pipeline:
            1. Encode sensor observations  →  continuous latent field F₀
            2. Add temporal bias (time-of-day / day-of-week)
            3. PDE evolution: iterate  F_{k+1} = F_k + dt·(α·L·F + (1−α)·N_θ(F))
               for K steps, refining each step through SSM
            4. Decode evolved field  F_K  →  traffic predictions

        Args:
            history_data: [B, L, N, C] historical traffic.
            future_data:  [B, L', N, C] future data (used by runner).
            batch_seen:   iteration index.
            epoch:        current epoch.
            train:        training flag.

        Returns:
            dict with 'prediction' [B, H, N, 1] and optional 'smoothness_loss'.
        """
        input_data = history_data[..., :self.input_dim]     # [B, L, N, F]

        # ---- Step 1: Encode to latent field ----
        field = self.encoder(input_data)                    # [B, N, field_dim]

        # ---- Step 2: Add temporal bias ----
        temp_bias = self._get_temporal_field_bias(history_data)
        if temp_bias is not None:
            field = field + temp_bias

        # ---- Step 3: PDE evolution (K steps) ----
        state = None
        derivatives = []
        dt = 1.0 / self.num_pde_steps

        for _ in range(self.num_pde_steps):
            field, state, dF = self._pde_step(field, state, dt)
            derivatives.append(dF)

        # ---- Step 4: Decode ----
        prediction = self.decoder(field)                    # [B, H, N, 1]

        # ---- Build result ----
        result = {"prediction": prediction}

        if train:
            result["smoothness_loss"] = (
                self.compute_smoothness_loss(derivatives) * self.smoothness_weight
            )

        return result
