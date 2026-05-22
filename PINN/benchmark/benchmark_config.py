"""Default paper benchmark settings."""

SEQ_LEN = 20
PRED_LEN = 10
ANCHOR_MODE = "last_valid"
DEGRADATION_LEVEL = "medium"
WINDOW_STRIDE = 1
EVAL_SPLIT = "test"
USE_CONTROL_AS_FEATURE = 0

PHYSICS_GUIDED_TRANSFORMER_MODELS = (
    "vanilla_transformer_ctrlfeat",
    "pgt_transformer_kin",
    "pgt_transformer_dyn_tau0",
    "pgt_transformer_dyn_controlled",
    "pgt_transformer_phys_controlled",
)

DEFAULT_PGT_LAMBDA_KIN = 0.05
DEFAULT_PGT_LAMBDA_DYN = 0.01
