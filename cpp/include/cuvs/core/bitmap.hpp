/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/bitmap.hpp>

namespace cuvs::core {
/* To use bitmap functions containing CUDA code, include <raft/core/bitmap.cuh> */

template <typename bitmap_t, typename index_t>  // NOLINT(readability-identifier-naming)
using bitmap_view = raft::core::bitmap_view<bitmap_t, index_t>;

}  // end namespace cuvs::core
