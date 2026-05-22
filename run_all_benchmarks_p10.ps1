$ErrorActionPreference = "Stop"

$env:AUV_PRED_LEN = "10"
$env:AUV_ANCHOR_POS_SOURCE = "last_valid"
$env:AUV_WINDOW_STRIDE = "1"
$env:AUV_DEGRADATION_LEVEL = "medium"
$env:AUV_USE_CONTROL_AS_FEATURE = "0"
$env:AUV_EVAL_SPLIT = "test"

$models = @(
    "cv",
    "lstm",
    "gru",
    "bilstm",
    "tcn",
    "vanilla_transformer",
    "vanilla_transformer_mask",
    "rope_transformer_nomask",
    "rope_transformer_mask_nophysics",
    "vrt_pinn_tau0",
    "vrt_pinn_controlled"
)

foreach ($model in $models) {
    Write-Host "============================================================"
    Write-Host "Running benchmark: $model"
    Write-Host "============================================================"
    $env:AUV_MODEL_NAME = $model

    if ($model -ne "cv") {
        python AUV_dataset\PINN\train.py
    }
    python AUV_dataset\PINN\evaluate.py
}

python AUV_dataset\PINN\benchmark\collect_results.py
