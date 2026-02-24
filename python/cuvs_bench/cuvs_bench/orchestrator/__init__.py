#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from .orchestrator import BenchmarkOrchestrator, run_benchmark
from .config_loaders import ConfigLoader, BenchmarkConfig, DatasetConfig, CppGBenchConfigLoader
from ..backends.registry import (
    get_backend_class,
    list_backends,
    register_config_loader,
    get_config_loader,
)

__all__ = [
    # Main orchestrator
    "BenchmarkOrchestrator",
    "run_benchmark",
    # Config loaders
    "ConfigLoader",
    "BenchmarkConfig",
    "DatasetConfig",
    "CppGBenchConfigLoader",
    # Registry functions
    "get_backend_class",
    "list_backends",
    "register_config_loader",
    "get_config_loader",
]


# ============================================================================
# Register built-in config loaders
# ============================================================================

def _register_builtin_loaders():
    """Register built-in config loaders."""
    register_config_loader("cpp_gbench", CppGBenchConfigLoader)


# Auto-register when module is imported
_register_builtin_loaders()
