/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUVS_ENABLE_JIT_LTO
#include "search_single_cta_kernel_launcher_jit.cuh"
#else
#include "search_single_cta_kernel_launcher.cuh"
#endif
