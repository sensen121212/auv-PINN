# =============================================================================
# export_onnx.py  —  Export RobustPINN to ONNX for Netron visualization
# =============================================================================
"""
Usage:
    cd d:\数学建模\total_matlab\会议\AUV_dataset\PINN
    python export_onnx.py

Output:
    robust_pinn.onnx  — Open with Netron (https://netron.app)
"""

import torch
from model import ModelConfig, RobustPINN

def main():
    # ---- 1. Build model with default config ----
    config = ModelConfig(
        n_features=12,
        seq_len=20,
        pred_len=5,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=256,
        dropout=0.0,       # Set to 0 for deterministic export
        n_dof=3,
    )

    model = RobustPINN(config)
    model.eval()

    # ---- 2. Create dummy inputs matching the forward() signature ----
    B = 1   # batch size
    T = config.seq_len   # 20
    F = config.n_features  # 12

    x_seq    = torch.randn(B, T, F)
    validity = torch.ones(B, T, 1)      # all frames valid
    last_pos = torch.randn(B, 3)

    # ---- 3. Export to ONNX ----
    output_path = "robust_pinn.onnx"

    torch.onnx.export(
        model,
        (x_seq, validity, last_pos),
        output_path,
        input_names=["x_seq", "validity", "last_pos"],
        output_names=["pred_pos"],
        dynamic_axes={
            "x_seq":    {0: "batch", 1: "seq_len"},
            "validity": {0: "batch", 1: "seq_len"},
            "last_pos": {0: "batch"},
            "pred_pos": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"[OK] ONNX model exported to: {output_path}")
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"     Input:  x_seq [{B}, {T}, {F}]  +  validity [{B}, {T}, 1]  +  last_pos [{B}, 3]")
    print(f"     Output: pred_pos [{B}, {config.pred_len}, 3]")
    print()
    print("Open with Netron:")
    print("  1. pip install netron  &&  netron robust_pinn.onnx")
    print("  2. Or drag-and-drop at https://netron.app")


if __name__ == "__main__":
    main()
