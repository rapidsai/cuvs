/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include "../core/exceptions.hpp"
#include <cuvs/selection/select_k.hpp>
#include <dlpack/dlpack.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

extern "C" cuvsError_t cuvsSelectK(cuvsResources_t res,
                                   DLManagedTensor* in_val,
                                   DLManagedTensor* out_val,
                                   DLManagedTensor* out_idx)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);

    int64_t n = in_val->dl_tensor.shape[1];
    int64_t k = out_val->dl_tensor.shape[1];

    auto in_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      static_cast<const float*>(in_val->dl_tensor.data), 1, n);

    auto out_val_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
      static_cast<float*>(out_val->dl_tensor.data), 1, k);

    auto out_idx_view = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
      static_cast<int64_t*>(out_idx->dl_tensor.data), 1, k);

    cuvs::selection::select_k(
      *res_ptr,
      in_view,
      std::nullopt,  // implicit positions [0, n) as in_idx
      out_val_view,
      out_idx_view,
      true);  // select_min = true (smallest distance = nearest neighbor)
  });
}
