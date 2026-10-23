/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cub_unique_launchers.hpp"

namespace cuvs::detail::unique_launch {
using compact32_pointer = void (*)(cub::ScanTileState<unsigned int>, int, unsigned int*);
using compact64_pointer = void (*)(cub::ScanTileState<unsigned long long>, int, unsigned long*);
}  // namespace cuvs::detail::unique_launch
template <>
CUB_RUNTIME_FUNCTION cuvs::detail::unique_launch::compact32_pointer
cuvs::detail::unique_launch::unique_0::CompactInitKernel();
template <>
CUB_RUNTIME_FUNCTION cuvs::detail::unique_launch::compact64_pointer
cuvs::detail::unique_launch::unique_1::CompactInitKernel();
