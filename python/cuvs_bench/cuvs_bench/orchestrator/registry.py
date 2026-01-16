#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Registry re-exports for orchestrator module.

This module re-exports registry functions from backends.registry for convenience.
The actual registry implementation lives in backends/registry.py.
"""

# Re-export only what's used
from ..backends.registry import (
    get_backend_class,
    list_backends,
    register_config_loader,
    get_config_loader,
)

__all__ = [
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
    from .config_loaders import CppGBenchConfigLoader
    register_config_loader("cpp_gbench", CppGBenchConfigLoader)


# Auto-register when module is imported
_register_builtin_loaders()
