"""Print the recommended benchmark order and commands."""

from __future__ import annotations

from benchmark_registry import MODEL_ORDER


def main() -> None:
    for name in MODEL_ORDER:
        if name == "cv":
            print(f"AUV_MODEL_NAME={name}: evaluate only")
        else:
            print(f"AUV_MODEL_NAME={name}: train.py then evaluate.py")


if __name__ == "__main__":
    main()
