"""Paper benchmark model registry."""

MODEL_ORDER = [
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
    "vrt_pinn_controlled",
]

GROUPS = {
    "cv": "Classical baseline",
    "lstm": "Generic data-driven baselines",
    "gru": "Generic data-driven baselines",
    "bilstm": "Generic data-driven baselines",
    "tcn": "Generic data-driven baselines",
    "vanilla_transformer": "Generic data-driven baselines",
    "vanilla_transformer_mask": "Method-specific ablations",
    "rope_transformer_nomask": "Method-specific ablations",
    "rope_transformer_mask_nophysics": "Method-specific ablations",
    "vrt_pinn_tau0": "Method-specific ablations",
    "vrt_pinn_controlled": "Method-specific ablations",
}
