/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

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
  // we're doing a physical merge here, make sure that this matches the merge_params
  RAFT_EXPECTS(params.merge_strategy == cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
               "cagra::merge only supports merge_strategy=MERGE_STRATEGY_PHYSICAL");

  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  std::size_t dim              = 0;
  std::size_t new_dataset_size = 0;
  int64_t stride               = -1;

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
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

  IdxT offset = 0;

  auto merge_dataset = [&](T* dst) {
    for (cagra_index_t* index : indices) {
      auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
      raft::copy_matrix(dst + offset * dim,
                        dim,
                        strided_dset->view().data_handle(),
                        static_cast<size_t>(stride),
                        dim,
                        static_cast<size_t>(strided_dset->n_rows()),
                        raft::resource::get_cuda_stream(handle));

      offset += IdxT(index->data().n_rows());
    }
  };

  cagra::index_params output_index_params = params.output_index_params;

  try {
    auto updated_dataset = raft::make_device_matrix<T, std::int64_t>(
      handle, std::int64_t(new_dataset_size), std::int64_t(dim));

    merge_dataset(updated_dataset.data_handle());

    auto merged_index =
      cagra::build(handle, output_index_params, raft::make_const_mdspan(updated_dataset.view()));
    if (!merged_index.data().is_owning() && output_index_params.attach_dataset_on_build) {
      using matrix_t           = decltype(updated_dataset);
      using layout_t           = typename matrix_t::layout_type;
      using container_policy_t = typename matrix_t::container_policy_type;
      using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
      auto out_layout          = raft::make_strided_layout(updated_dataset.view().extents(),
                                                  std::array<int64_t, 2>{stride, 1});
      merged_index.update_dataset(handle, owning_t{std::move(updated_dataset), out_layout});
    }
    RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
    return merged_index;

  } catch (std::bad_alloc& e) {
    RAFT_LOG_DEBUG("cagra::merge: using host memory for merged dataset");

    auto updated_dataset =
      raft::make_host_matrix<T, std::int64_t>(std::int64_t(new_dataset_size), std::int64_t(dim));

    merge_dataset(updated_dataset.data_handle());

    auto merged_index =
      cagra::build(handle, output_index_params, raft::make_const_mdspan(updated_dataset.view()));
    if (!merged_index.data().is_owning() && output_index_params.attach_dataset_on_build) {
      using matrix_t           = decltype(updated_dataset);
      using layout_t           = typename matrix_t::layout_type;
      using container_policy_t = typename matrix_t::container_policy_type;
      using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
      auto out_layout          = raft::make_strided_layout(updated_dataset.view().extents(),
                                                  std::array<int64_t, 2>{stride, 1});
      merged_index.update_dataset(handle, owning_t{std::move(updated_dataset), out_layout});
    }
    return merged_index;
  }
}

}  // namespace cuvs::neighbors::cagra::detail
