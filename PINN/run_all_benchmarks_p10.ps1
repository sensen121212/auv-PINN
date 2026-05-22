$ErrorActionPreference = "Stop"

$env:AUV_PRED_LEN = "10"
$env:AUV_ANCHOR_POS_SOURCE = "last_valid"
$env:AUV_WINDOW_STRIDE = "1"
$env:AUV_DEGRADATION_LEVEL = "medium"
$env:AUV_USE_CONTROL_AS_FEATURE = "0"
$env:AUV_EVAL_SPLIT = "test"
$env:AUV_PHYSICS_MODE = "none"
$env:AUV_CONTROL_MODE = "none"
$env:AUV_LAMBDA_KIN = "0.05"
$env:AUV_LAMBDA_DYN = "0.01"

$models = @(
    "cv",
    "lstm",
    "gru",
    "bilstm",
    "tcn",
    "vanilla_transformer",
    "vanilla_transformer_ctrlfeat",
    "pgt_transformer_kin",
    "pgt_transformer_dyn_tau0",
    "pgt_transformer_dyn_controlled",
    "pgt_transformer_phys_controlled",
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
    $env:AUV_USE_CONTROL_AS_FEATURE = "0"
    $env:AUV_PHYSICS_MODE = "none"
    $env:AUV_CONTROL_MODE = "none"
    $env:AUV_LAMBDA_KIN = "0.05"
    $env:AUV_LAMBDA_DYN = "0.01"

    switch ($model) {
        "vanilla_transformer_ctrlfeat" {
            $env:AUV_USE_CONTROL_AS_FEATURE = "1"
        }
        "pgt_transformer_kin" {
            $env:AUV_USE_CONTROL_AS_FEATURE = "1"
            $env:AUV_LAMBDA_KIN = "0.05"
        }
        "pgt_transformer_dyn_tau0" {
            $env:AUV_PHYSICS_MODE = "inference"
            $env:AUV_CONTROL_MODE = "none"
            $env:AUV_LAMBDA_DYN = "0.01"
        }
        "pgt_transformer_dyn_controlled" {
            $env:AUV_USE_CONTROL_AS_FEATURE = "1"
            $env:AUV_PHYSICS_MODE = "inference"
            $env:AUV_CONTROL_MODE = "anchor_hold"
            $env:AUV_LAMBDA_DYN = "0.01"
        }
        "pgt_transformer_phys_controlled" {
            $env:AUV_USE_CONTROL_AS_FEATURE = "1"
            $env:AUV_PHYSICS_MODE = "inference"
            $env:AUV_CONTROL_MODE = "anchor_hold"
            $env:AUV_LAMBDA_KIN = "0.05"
            $env:AUV_LAMBDA_DYN = "0.01"
        }
        "vrt_pinn_tau0" {
            $env:AUV_PHYSICS_MODE = "inference"
            $env:AUV_CONTROL_MODE = "none"
        }
        "vrt_pinn_controlled" {
            $env:AUV_PHYSICS_MODE = "inference"
            $env:AUV_CONTROL_MODE = "anchor_hold"
        }
    }

    if ($model -ne "cv") {
        python AUV_dataset\PINN\train.py
    }
    python AUV_dataset\PINN\evaluate.py
}

python AUV_dataset\PINN\benchmark\collect_results.py
