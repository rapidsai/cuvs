/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
namespace cuvs::detail {
template <typename InType, typename OutType>
void sort_cols_per_row(const InType* in,
                       OutType* out,
                       int n_rows,
                       int n_columns,
                       bool& bAllocWorkspace,
                       void* workspacePtr,
                       size_t& workspaceSize,
                       cudaStream_t stream,
                       InType* sortedKeys = nullptr);
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 3>>
void sort_cols_per_row(Args... args);
}  // namespace cuvs::detail
