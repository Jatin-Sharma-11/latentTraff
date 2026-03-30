#!/usr/bin/env bash
# =============================================================================
# Run all CLFMv2 ablation experiments on PEMS08
# =============================================================================
# Usage:
#   bash idea/CLFM/v2/ablations/run_all_ablations.sh [GPU_ID]
#
# Each experiment removes exactly one component from the full CLFMv2 model.
# Results are saved to separate checkpoint directories.
# =============================================================================

GPU=${1:-0}
WORKSPACE=$(cd "$(dirname "$0")/../../../.." && pwd)

CONFIGS=(
    "idea/CLFM/v2/ablations/PEMS08_no_laplacian.py"
    "idea/CLFM/v2/ablations/PEMS08_no_neural_pde.py"
    "idea/CLFM/v2/ablations/PEMS08_no_ssm.py"
    "idea/CLFM/v2/ablations/PEMS08_no_temporal_emb.py"
    "idea/CLFM/v2/ablations/PEMS08_no_spatial_coords.py"
    "idea/CLFM/v2/ablations/PEMS08_no_smoothness_loss.py"
)

NAMES=(
    "NoLaplacian"
    "NoNeuralPDE"
    "NoSSM"
    "NoTemporalEmb"
    "NoSpatialCoords"
    "NoSmoothnessLoss"
)

cd "$WORKSPACE"

echo "========================================"
echo " CLFMv2 Ablation Study — PEMS08"
echo " GPU: $GPU"
echo " Workspace: $WORKSPACE"
echo "========================================"

for i in "${!CONFIGS[@]}"; do
    echo ""
    echo "──────────────────────────────────────"
    echo " [$((i+1))/${#CONFIGS[@]}]  ${NAMES[$i]}"
    echo " Config: ${CONFIGS[$i]}"
    echo "──────────────────────────────────────"
    python experiments/train.py -c "${CONFIGS[$i]}" --gpus "$GPU"
    echo " ✓ ${NAMES[$i]} complete"
done

echo ""
echo "========================================"
echo " All ablation experiments finished."
echo "========================================"
