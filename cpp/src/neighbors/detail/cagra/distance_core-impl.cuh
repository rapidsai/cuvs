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
#pragma once

#include <raft/core/logger-macros.hpp>

#include "compute_distance.hpp"

namespace cuvs::neighbors::cagra::detail {

// template <uint32_t TeamSize,
//           uint32_t DatasetBlockDim,
//           typename DataT,
//           typename IndexT,
//           typename DistanceT>
// __launch_bounds__(1, 1) __global__ void standard_dataset_descriptor_init_kernel(
//   dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
//   const DataT* ptr,
//   IndexT size,
//   uint32_t dim,
//   size_t ld)
// {
//   new (out) standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>(
//     ptr, size, dim, ld);
//   (void)out->set_smem_ws(out);
// }

}  // namespace cuvs::neighbors::cagra::detail
