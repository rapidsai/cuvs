#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Elasticsearch GPU backend plugin for cuvs-bench.

Install with: pip install cuvs-bench[elastic]
"""

from .backend import register

__all__ = ["register"]
