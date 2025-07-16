
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/distance/distance.hpp>

namespace {

template <typename T, typename DistT, typename LayoutT = raft::row_major>
void _pairwise_distance(cuvsResources_t res,
                        DLManagedTensor* x_tensor,
                        DLManagedTensor* y_tensor,
                        DLManagedTensor* distances_tensor,
                        cuvsDistanceType metric,
                        float metric_arg)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  using mdspan_type           = raft::device_matrix_view<T const, int64_t, LayoutT>;
  using distances_mdspan_type = raft::device_matrix_view<DistT, int64_t, LayoutT>;

  auto x_mds         = cuvs::core::from_dlpack<mdspan_type>(x_tensor);
  auto y_mds         = cuvs::core::from_dlpack<mdspan_type>(y_tensor);
  auto distances_mds = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::distance::pairwise_distance(*res_ptr, x_mds, y_mds, distances_mds, metric, metric_arg);
}
}  // namespace

extern "C" cuvsError_t cuvsPairwiseDistance(cuvsResources_t res,
                                            DLManagedTensor* x_tensor,
                                            DLManagedTensor* y_tensor,
                                            DLManagedTensor* distances_tensor,
                                            cuvsDistanceType metric,
                                            float metric_arg)
{
  return cuvs::core::translate_exceptions([=] {
    auto x_dt    = x_tensor->dl_tensor.dtype;
    auto y_dt    = x_tensor->dl_tensor.dtype;
    auto dist_dt = x_tensor->dl_tensor.dtype;

    // Type validation based on metric
    if (metric == cuvsDistanceType::BitwiseHamming) {
      // BitwiseHamming requires integer types
      if ((x_dt.code != kDLUInt) || (y_dt.code != kDLUInt) || (dist_dt.code != kDLUInt)) {
        RAFT_FAIL("BitwiseHamming metric requires unsigned integer tensors");
      }
    } else {
      // All other metrics require floating point types
      if ((x_dt.code != kDLFloat) || (y_dt.code != kDLFloat) || (dist_dt.code != kDLFloat)) {
        RAFT_FAIL("Inputs to cuvsPairwiseDistance must all be floating point tensors (except BitwiseHamming which requires unsigned integers)");
      }
    }

    if ((x_dt.bits != y_dt.bits) || (x_dt.bits != dist_dt.bits)) {
      RAFT_FAIL("Inputs to cuvsPairwiseDistance must all have the same dtype");
    }

    bool x_row_major;
    if (cuvs::core::is_c_contiguous(x_tensor)) {
      x_row_major = true;
    } else if (cuvs::core::is_f_contiguous(x_tensor)) {
      x_row_major = false;
    } else {
      RAFT_FAIL("X input to cuvsPairwiseDistance must be contiguous (non-strided)");
    }

    bool y_row_major;
    if (cuvs::core::is_c_contiguous(y_tensor)) {
      y_row_major = true;
    } else if (cuvs::core::is_f_contiguous(y_tensor)) {
      y_row_major = false;
    } else {
      RAFT_FAIL("Y input to cuvsPairwiseDistance must be contiguous (non-strided)");
    }

    bool distances_row_major;
    if (cuvs::core::is_c_contiguous(distances_tensor)) {
      distances_row_major = true;
    } else if (cuvs::core::is_f_contiguous(distances_tensor)) {
      distances_row_major = false;
    } else {
      RAFT_FAIL("distances input to cuvsPairwiseDistance must be contiguous (non-strided)");
    }

    if ((x_row_major != y_row_major) || (x_row_major != distances_row_major)) {
      RAFT_FAIL(
        "Inputs to cuvsPairwiseDistance must all have the same layout (row-major or col-major)");
    }

    if (x_row_major) {
      if (metric == cuvsDistanceType::BitwiseHamming) {
        // Only uint8_t supported for BitwiseHamming (with internal optimization)
        if (x_dt.bits == 8) {
          _pairwise_distance<uint8_t, uint32_t>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else {
          RAFT_FAIL("BitwiseHamming only supports 8-bit unsigned integer input. Received: %d bits", x_dt.bits);
        }
      } else {
        // Float types for other metrics
        if (x_dt.bits == 32) {
          _pairwise_distance<float, float>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else if (x_dt.bits == 16) {
          _pairwise_distance<half, float>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else if (x_dt.bits == 64) {
          _pairwise_distance<double, double>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else {
          RAFT_FAIL("Unsupported DLtensor dtype: %d and bits: %d", x_dt.code, x_dt.bits);
        }
      }
    } else {
      if (metric == cuvsDistanceType::BitwiseHamming) {
        // Only uint8_t supported for BitwiseHamming (column major)
        if (x_dt.bits == 8) {
          _pairwise_distance<uint8_t, uint32_t, raft::col_major>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else {
          RAFT_FAIL("BitwiseHamming only supports 8-bit unsigned integer input. Received: %d bits", x_dt.bits);
        }
      } else {
        // Float types for other metrics (column major)
        if (x_dt.bits == 32) {
          _pairwise_distance<float, float, raft::col_major>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else if (x_dt.bits == 16) {
          _pairwise_distance<half, float, raft::col_major>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else if (x_dt.bits == 64) {
          _pairwise_distance<double, double, raft::col_major>(
            res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
        } else {
          RAFT_FAIL("Unsupported DLtensor dtype: %d and bits: %d", x_dt.code, x_dt.bits);
        }
      }
    }
  });
}
