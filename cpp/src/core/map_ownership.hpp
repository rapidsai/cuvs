/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/map.cuh>

namespace cuvs::spatial::knn::detail::utils {
template <typename T>
struct mapping;
}

// This also covers named maps instantiated inside RAFT operations. Owning the
// host wrapper leaves CUDA kernel pointers and launches in the same TU.
#define CUVS_MAP_INSTANTIATION(...) extern template __VA_ARGS__;
#include "map_instantiations.inl"
#undef CUVS_MAP_INSTANTIATION
