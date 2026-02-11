/**
 * SPDX-FileCopyrightText: Copyright (c) Facebook, Inc. and its affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

namespace cuvs::neighbors::detail::faiss_select {

template <typename T>
struct comparator {
  __device__ static inline auto lt(T a, T b) -> bool { return a < b; }

  __device__ static inline auto gt(T a, T b) -> bool { return a > b; }
};

template <>
struct comparator<half> {
  __device__ static inline auto lt(half a, half b) -> bool { return __hlt(a, b); }

  __device__ static inline auto gt(half a, half b) -> bool { return __hgt(a, b); }
};

}  // namespace cuvs::neighbors::detail::faiss_select
