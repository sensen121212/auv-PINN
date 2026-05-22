"""Collect benchmark JSON outputs into CSV/Markdown paper tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PINN_ROOT = THIS_DIR.parent
if str(PINN_ROOT) not in sys.path:
    sys.path.insert(0, str(PINN_ROOT))

from config import Config
from benchmark.benchmark_registry import MODEL_ORDER


MAIN_COLUMNS = [
    "Group",
    "Model",
    "ModelName",
    "Params",
    "RMSE",
    "MAE",
    "ADE",
    "FDE",
    "Degraded_RMSE",
    "Normal_RMSE",
    "High_Maneuver_RMSE",
    "LastStep_RMSE",
    "FossenResidual_norm",
    "TrajectoryAccelNorm",
    "BestValMSE",
    "BestEpoch",
    "TestSplit",
]


def _load_results(cfg: Config, split: str) -> List[Dict[str, object]]:
    result_dir = PINN_ROOT / "benchmark" / "results"
    rows: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        path = result_dir / (
            f"result_p{cfg.PRED_LEN}_anchor_{cfg.ANCHOR_POS_SOURCE}"
            f"_deg_{cfg.DEGRADATION_LEVEL}_{model_name}_{split}.json"
        )
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        per_step = payload.get("per_step_rmse", {})
        row = {
            "Group": payload.get("Group"),
            "Model": payload.get("Model"),
            "ModelName": payload.get("ModelName"),
            "Params": payload.get("Params"),
            "BestValMSE": payload.get("BestValMSE"),
            "BestEpoch": payload.get("BestEpoch"),
            "TestSplit": payload.get("TestSplit"),
        }
        row.update(metrics)
        pred_len = int(payload.get("PredLen", cfg.PRED_LEN))
        row["LastStep_RMSE"] = per_step.get(f"step_{pred_len}")
        rows.append(row)
    return rows


def _load_long_table(cfg: Config, split: str, key: str) -> pd.DataFrame:
    result_dir = PINN_ROOT / "benchmark" / "results"
    rows: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        path = result_dir / (
            f"result_p{cfg.PRED_LEN}_anchor_{cfg.ANCHOR_POS_SOURCE}"
            f"_deg_{cfg.DEGRADATION_LEVEL}_{model_name}_{split}.json"
        )
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(key, {})
        for name, value in values.items():
            rows.append({
                "Group": payload.get("Group"),
                "Model": payload.get("Model"),
                "ModelName": payload.get("ModelName"),
                "Bin" if key == "anchor_lag_rmse" else "Step": name,
                "RMSE": value,
            })
    return pd.DataFrame(rows)


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a small GitHub-flavored Markdown table without extra deps."""
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(
            lambda v: "" if pd.isna(v) else (f"{v:.6g}" if isinstance(v, float) else str(v))
        )
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in text_df.columns) + " |"
        for _, row in text_df.iterrows()
    ]
    return "\n".join([header, sep] + rows) + "\n"


def main() -> None:
    cfg = Config()
    split = "test"
    rows = _load_results(cfg, split)
    out_dir = PINN_ROOT / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No benchmark result JSON files found.")
        return

    df = pd.DataFrame(rows)
    for col in MAIN_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[MAIN_COLUMNS]

    csv_path = out_dir / f"benchmark_results_p{cfg.PRED_LEN}.csv"
    md_path = out_dir / f"benchmark_results_p{cfg.PRED_LEN}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(_to_markdown(df), encoding="utf-8")

    per_step = _load_long_table(cfg, split, "per_step_rmse")
    per_step_path = out_dir / f"benchmark_per_step_p{cfg.PRED_LEN}.csv"
    per_step.to_csv(per_step_path, index=False, encoding="utf-8-sig")

    anchor_lag = _load_long_table(cfg, split, "anchor_lag_rmse")
    anchor_path = out_dir / f"benchmark_anchor_lag_p{cfg.PRED_LEN}.csv"
    anchor_lag.to_csv(anchor_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {per_step_path}")
    print(f"Wrote {anchor_path}")


if __name__ == "__main__":
    main()
