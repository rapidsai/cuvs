/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/kvp.hpp>
#include <raft/linalg/detail/coalesced_reduction.cuh>
#include <raft/linalg/detail/norm.cuh>
#include <raft/linalg/detail/strided_reduction.cuh>

// Own the existing host launchers, including the partial reduction shared by
// thick reductions with different input operations. Kernel pointers and their
// launches stay together; callers retain their original validation and math.
#define CUVS_REDUCTION_INSTANTIATION(...) extern template __VA_ARGS__;
#include "reduction_instantiations.inl"
#undef CUVS_REDUCTION_INSTANTIATION
