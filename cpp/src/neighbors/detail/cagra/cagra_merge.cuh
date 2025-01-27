/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../../../core/nvtx.hpp"
#include "../../vpq_dataset.cuh"
#include "cagra_build.cuh"
#include "graph_core.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <cuvs/neighbors/nn_descent.hpp>

// TODO: This shouldn't be calling spatial/knn APIs
#include "../ann_utils.cuh"

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

template <class T, class IdxT>
index<T, IdxT> merge(raft::resources const& handle,
                     const cagra::merge_params& params,
                     std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices)
{
  std::size_t dim              = 0;
  std::size_t new_dataset_size = 0;
  int64_t stride               = -1;

  for (auto index : indices) {
    using ds_idx_type = decltype(index->data().n_rows());
    if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
        strided_dset != nullptr) {
      if (dim == 0) {
        dim    = index->dim();
        stride = strided_dset->stride();
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else if (dynamic_cast<const cuvs::neighbors::empty_dataset<int64_t>*>(&index->data()) !=
               nullptr) {
      RAFT_FAIL(
        "cagra::merge only supports an index to which the dataset is attached. Please check if the "
        "index was built with index_param.attach_dataset_on_build = true, or if a dataset was "
        "attached after the build.");
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dataset index");
    }
  }

  auto host_updated_dataset = raft::make_host_matrix<T, std::int64_t>(new_dataset_size, dim);
  memset(host_updated_dataset.data_handle(), 0, sizeof(T) * host_updated_dataset.size());

  IdxT offset = 0;

  for (auto index : indices) {
    using ds_idx_type  = decltype(index->data().n_rows());
    auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());

    RAFT_CUDA_TRY(cudaMemcpy2DAsync(host_updated_dataset.data_handle() + offset * dim,
                                    sizeof(T) * dim,
                                    strided_dset->view().data_handle(),
                                    sizeof(T) * stride,
                                    sizeof(T) * dim,
                                    strided_dset->n_rows(),
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(handle)));

    offset += IdxT(index->data().n_rows());
  }
  // Allocate the new dataset on device
  auto device_updated_dataset =
    raft::make_device_matrix<T, std::int64_t>(handle, new_dataset_size, dim);
  auto device_updated_dataset_view = raft::make_device_matrix_view<T, std::int64_t>(
    device_updated_dataset.data_handle(), new_dataset_size, dim);

  // Copy updated dataset on host memory to device memory
  raft::copy(device_updated_dataset.data_handle(),
             host_updated_dataset.data_handle(),
             new_dataset_size * dim,
             raft::resource::get_cuda_stream(handle));

  auto merged_index =
    cagra::build(handle, params, raft::make_const_mdspan(device_updated_dataset_view));

  if (static_cast<std::size_t>(stride) == dim) {
    using out_mdarray_type          = decltype(device_updated_dataset);
    using out_layout_type           = typename out_mdarray_type::layout_type;
    using out_container_policy_type = typename out_mdarray_type::container_policy_type;
    using out_owning_type = owning_dataset<T, int64_t, out_layout_type, out_container_policy_type>;
    auto out_layout       = raft::make_strided_layout(device_updated_dataset_view.extents(),
                                                std::array<int64_t, 2>{stride, 1});

    merged_index.update_dataset(handle,
                                out_owning_type{std::move(device_updated_dataset), out_layout});
  }
  return merged_index;
}

}  // namespace cuvs::neighbors::cagra::detail
