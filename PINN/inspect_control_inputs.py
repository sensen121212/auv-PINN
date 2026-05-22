"""Audit control-input availability for controlled Fossen dynamics.

The script checks processed PINN CSV files and raw trajectory CSV files for
propulsion/control columns, then writes a JSON and Markdown report under
``AUV_dataset/output``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PINN_DIR = Path(__file__).resolve().parent
DATA_ROOT = PINN_DIR.parent
OUTPUT_DIR = DATA_ROOT / "output"

PRIMARY_FIELDS = ["thrust_net_N", "rudder_rad", "stern_rad"]
CANDIDATE_FIELDS = [
    "thrust_net_N", "rudder_rad", "stern_rad",
    "thrust", "prop_force", "propeller_force", "propulsion_force",
    "rpm", "motor_rpm", "prop_rpm",
    "rudder", "rudder_cmd", "rudder_angle",
    "stern", "stern_cmd", "stern_angle",
    "elevator", "elevator_angle", "fin", "fin_angle",
    "control", "tau", "X_force", "Y_force", "Z_force",
    "K_moment", "M_moment", "N_moment",
    "Thrust_net_N", "delta_r", "delta_s", "n_prop",
]


def _read_columns(path: Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _case_insensitive_matches(columns: Iterable[str], candidates: Iterable[str]) -> Dict[str, str]:
    lower_to_original = {c.lower(): c for c in columns}
    matches: Dict[str, str] = {}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_to_original:
            matches[candidate] = lower_to_original[key]
    return matches


def _column_stats(path: Path, columns: List[str]) -> Dict[str, Dict[str, object]]:
    if not columns:
        return {}

    stats: Dict[str, Dict[str, object]] = {}
    data = pd.read_csv(path, usecols=columns)
    n = len(data)
    for col in columns:
        numeric = pd.to_numeric(data[col], errors="coerce")
        non_null = numeric.notna()
        arr = numeric[non_null].to_numpy(dtype=np.float64)
        if arr.size == 0:
            stats[col] = {
                "non_null_rate": 0.0,
                "non_zero_rate": 0.0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "unique_count": 0,
                "almost_constant": True,
            }
            continue

        unique_count = int(pd.Series(arr).nunique(dropna=True))
        std = float(np.std(arr))
        stats[col] = {
            "non_null_rate": float(non_null.mean()) if n else 0.0,
            "non_zero_rate": float(np.mean(np.abs(arr) > 1e-12)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": std,
            "unique_count": unique_count,
            "almost_constant": bool(unique_count <= 1 or std < 1e-9),
        }
    return stats


def _audit_csv(path: Path, stat_all_matches: bool = True) -> Dict[str, object]:
    columns = _read_columns(path)
    matches = _case_insensitive_matches(columns, CANDIDATE_FIELDS)
    present_primary = {field: field in columns for field in PRIMARY_FIELDS}
    stat_columns = list(matches.values()) if stat_all_matches else [
        field for field in PRIMARY_FIELDS if field in columns
    ]
    sample = pd.read_csv(path, nrows=5)

    return {
        "path": str(path),
        "columns": columns,
        "primary_present": present_primary,
        "candidate_matches": matches,
        "stats": _column_stats(path, sorted(set(stat_columns))),
        "sample_rows": sample.to_dict(orient="records"),
    }


def _find_raw_csvs() -> List[Path]:
    roots = [
        DATA_ROOT / "trajectory_dataset",
        DATA_ROOT,
    ]
    raw_files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if OUTPUT_DIR in path.parents:
                continue
            if path.name == "trajectory_splits.csv":
                continue
            raw_files.append(path)
    return sorted(set(raw_files))


def _audit_raw_files(limit: Optional[int] = None) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    files = _find_raw_csvs()
    if limit is not None:
        files = files[:limit]
    for path in files:
        try:
            columns = _read_columns(path)
        except Exception as exc:
            reports.append({"path": str(path), "error": str(exc)})
            continue
        matches = _case_insensitive_matches(columns, CANDIDATE_FIELDS)
        if matches:
            reports.append({
                "path": str(path),
                "columns": columns,
                "candidate_matches": matches,
                "stats": _column_stats(path, sorted(set(matches.values()))),
                "sample_rows": pd.read_csv(path, nrows=3).to_dict(orient="records"),
            })
    return reports


def _controlled_dynamics_conclusion(processed_reports: List[Dict[str, object]]) -> str:
    primary_ok = all(
        report["primary_present"].get(field, False)
        for report in processed_reports
        for field in PRIMARY_FIELDS
    )
    if primary_ok:
        return (
            "SUPPORTED: processed CSVs contain thrust_net_N, rudder_rad, "
            "and stern_rad. Controlled Fossen residual can use these columns."
        )

    any_rpm = any(
        any("rpm" in key.lower() for key in report["candidate_matches"].keys())
        for report in processed_reports
    )
    if any_rpm:
        return (
            "PARTIAL: RPM-like fields exist but thrust_net_N is missing. A "
            "propulsor mapping T=f(rpm, inflow) is required; no coefficients "
            "should be invented in this audit."
        )

    any_surface = any(
        any(key.lower() in {"rudder", "rudder_cmd", "rudder_angle", "stern", "stern_cmd", "stern_angle", "elevator", "elevator_angle"} for key in report["candidate_matches"].keys())
        for report in processed_reports
    )
    if any_surface:
        return (
            "PARTIAL: control-surface angles exist. They may be mapped to "
            "pitch/yaw moments via learnable control-effectiveness coefficients, "
            "but propulsion force is unavailable."
        )

    return (
        "NOT SUPPORTED: processed CSVs do not contain propulsion/control "
        "columns. tau≈0 is only a passive regularizer unless data generation "
        "exports controls."
    )


def _write_markdown(report: Dict[str, object], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Control Input Audit")
    lines.append("")
    lines.append(f"Conclusion: {report['conclusion']}")
    lines.append("")
    lines.append("## Processed CSVs")
    for item in report["processed_csvs"]:
        lines.append(f"### {Path(item['path']).name}")
        lines.append("")
        lines.append("Columns:")
        lines.append(", ".join(item["columns"]))
        lines.append("")
        lines.append("Primary fields:")
        for field, present in item["primary_present"].items():
            lines.append(f"- {field}: {present}")
        lines.append("")
        lines.append("Stats:")
        for col, stats in item["stats"].items():
            lines.append(
                f"- {col}: non_null={stats['non_null_rate']:.6f}, "
                f"non_zero={stats['non_zero_rate']:.6f}, "
                f"min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']}, std={stats['std']}, "
                f"unique={stats['unique_count']}, "
                f"almost_constant={stats['almost_constant']}"
            )
        lines.append("")
        lines.append("Sample rows:")
        lines.append("```json")
        lines.append(json.dumps(item["sample_rows"][:3], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Raw CSVs With Control-Like Fields")
    if not report["raw_csvs_with_control_candidates"]:
        lines.append("No raw CSV with control-like candidate fields was found.")
    for item in report["raw_csvs_with_control_candidates"]:
        lines.append(f"### {item['path']}")
        lines.append(f"Matches: {item['candidate_matches']}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    processed_paths = [
        OUTPUT_DIR / "ground_truth.csv",
        OUTPUT_DIR / "corrupted_data.csv",
    ]
    processed_reports = [_audit_csv(path) for path in processed_paths if path.exists()]
    report = {
        "processed_csvs": processed_reports,
        "raw_csvs_with_control_candidates": _audit_raw_files(),
        "conclusion": _controlled_dynamics_conclusion(processed_reports),
        "notes": {
            "rpm_mapping": "If only RPM is available, define thrust_net_N = f(rpm, inflow) from REMUS/propulsor calibration before using controlled dynamics.",
            "surface_mapping": "Rudder/stern angles can drive yaw/pitch moments through learnable coefficients already present in FossenDynamicsLoss.",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "control_input_audit.json"
    md_path = OUTPUT_DIR / "control_input_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)

    print("=" * 80)
    print("Control Input Audit")
    print("=" * 80)
    print(report["conclusion"])
    for item in processed_reports:
        print(f"\n[{Path(item['path']).name}]")
        print("Columns:", ", ".join(item["columns"]))
        for field, present in item["primary_present"].items():
            print(f"  {field}: {present}")
        for col, stats in item["stats"].items():
            print(
                f"  {col}: non_null={stats['non_null_rate']:.4f}, "
                f"non_zero={stats['non_zero_rate']:.4f}, "
                f"min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']}, std={stats['std']}, "
                f"unique={stats['unique_count']}, "
                f"almost_constant={stats['almost_constant']}"
            )
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
