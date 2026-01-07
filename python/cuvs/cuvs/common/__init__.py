# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


from .mg_resources import MultiGpuResources, auto_sync_multi_gpu_resources
from .resources import Resources, auto_sync_resources

__all__ = [
    "auto_sync_resources",
    "Resources",
    "MultiGpuResources",
    "auto_sync_multi_gpu_resources",
]
