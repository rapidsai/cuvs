
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

template <typename T>
void _pairwise_distance(cuvsResources_t res,
                        DLManagedTensor* x_tensor,
                        DLManagedTensor* y_tensor,
                        DLManagedTensor* distances_tensor,
                        cuvsDistanceType metric,
                        float metric_arg)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  using mdspan_type           = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<T, int64_t, raft::row_major>;

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

    if ((x_dt.code != kDLFloat) || (y_dt.code != kDLFloat) || (dist_dt.code != kDLFloat)) {
      RAFT_FAIL("Inputs to cuvsPairwiseDistance must all be floating point tensors");
    }

    if ((x_dt.bits != y_dt.bits) || (x_dt.bits != dist_dt.bits)) {
      RAFT_FAIL("Inputs to cuvsPairwiseDistance must all have the same dtype");
    }

    if (x_dt.bits == 32) {
      _pairwise_distance<float>(res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
    } else if (x_dt.bits == 64) {
      _pairwise_distance<double>(res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
    } else {
      RAFT_FAIL("Unsupported DLtensor dtype: %d and bits: %d", x_dt.code, x_dt.bits);
    }
  });
}
