"""Benchmark baseline models for AUV short-horizon trajectory prediction."""

from .baseline_factory import (
    BENCHMARK_MODEL_NAMES,
    BenchmarkModelSpec,
    create_benchmark_model,
    get_benchmark_spec,
    is_data_driven_model,
    is_physics_guided_model,
    is_pinn_model,
)

__all__ = [
    "BENCHMARK_MODEL_NAMES",
    "BenchmarkModelSpec",
    "create_benchmark_model",
    "get_benchmark_spec",
    "is_data_driven_model",
    "is_physics_guided_model",
    "is_pinn_model",
]
