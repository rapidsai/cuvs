/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>
#include <raft/matrix/detail/select_k-inl.cuh>
#include <raft/sparse/matrix/detail/select_k-inl.cuh>

// Keep shared host launchers and their kernel launches in one owning TU.
#define CUVS_SELECTION_INSTANTIATION(...) extern template __VA_ARGS__;
#include "selection_instantiations.inl"
#undef CUVS_SELECTION_INSTANTIATION
