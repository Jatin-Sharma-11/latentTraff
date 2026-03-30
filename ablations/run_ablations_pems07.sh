#!/usr/bin/env bash
# Run all CLFMv2 ablation experiments on PEMS07 (50 epochs)
GPU=${1:-0}
WORKSPACE=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$WORKSPACE"

CONFIGS=(
    "idea/CLFM/v2/ablations/PEMS07_no_laplacian.py"
    "idea/CLFM/v2/ablations/PEMS07_no_neural_pde.py"
    "idea/CLFM/v2/ablations/PEMS07_no_ssm.py"
    "idea/CLFM/v2/ablations/PEMS07_no_temporal_emb.py"
    "idea/CLFM/v2/ablations/PEMS07_no_spatial_coords.py"
    "idea/CLFM/v2/ablations/PEMS07_no_smoothness_loss.py"
)

NAMES=(
    "NoLaplacian"
    "NoNeuralPDE"
    "NoSSM"
    "NoTemporalEmb"
    "NoSpatialCoords"
    "NoSmoothnessLoss"
)

echo "========================================"
echo " CLFMv2 Ablation Study — PEMS07 (50 ep)"
echo " GPU: $GPU | Workspace: $WORKSPACE"
echo "========================================"

for i in "${!CONFIGS[@]}"; do
    echo ""
    echo "── [$((i+1))/${#CONFIGS[@]}] ${NAMES[$i]} ──"
    python experiments/train.py -c "${CONFIGS[$i]}" --gpus "$GPU"
    echo " ✓ ${NAMES[$i]} done"
done

echo ""
echo "========================================"
echo " PEMS07 ablations finished."
echo "========================================"
