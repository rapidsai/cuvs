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

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvs::bench::ann {

template <typename T, typename IdxT>
class CuvsIvfFlatGpu : public ANN<T>, public AnnGPU {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    cuvs::neighbors::ivf_flat::search_params ivf_flat_params;
  };

  using BuildParam = cuvs::neighbors::ivf_flat::index_params;

  CuvsIvfFlatGpu(Metric metric, int dim, const BuildParam& param)
    : ANN<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric                         = parse_metric_type(metric);
    index_params_.conservative_memory_allocation = true;
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const AnnSearchParam& param) override;

  void search(const T* queries,
              int batch_size,
              int k,
              AnnBase::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  // to enable dataset access from GPU memory
  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::HostMmap;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<ANN<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  BuildParam index_params_;
  cuvs::neighbors::ivf_flat::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::ivf_flat::index<T, IdxT>> index_;
  int device_;
  int dimension_;
};

template <typename T, typename IdxT>
void CuvsIvfFlatGpu<T, IdxT>::build(const T* dataset, size_t nrow)
{
  index_ = std::make_shared<cuvs::neighbors::ivf_flat::index<T, IdxT>>(
    std::move(cuvs::neighbors::ivf_flat::build(
      handle_,
      index_params_,
      raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, dimension_))));
  // Note: internally the IVF-Flat build works with simple pointers, and accepts both host and
  // device pointer. Therefore, although we provide here a host_mdspan, this works with device
  // pointer too.
}

template <typename T, typename IdxT>
void CuvsIvfFlatGpu<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.ivf_flat_params;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void CuvsIvfFlatGpu<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::ivf_flat::serialize_file(handle_, file, *index_);
  return;
}

template <typename T, typename IdxT>
void CuvsIvfFlatGpu<T, IdxT>::load(const std::string& file)
{
  index_ =
    std::make_shared<cuvs::neighbors::ivf_flat::index<T, IdxT>>(handle_, index_params_, this->dim_);

  cuvs::neighbors::ivf_flat::deserialize_file(handle_, file, index_.get());
  return;
}

template <typename T, typename IdxT>
std::unique_ptr<ANN<T>> CuvsIvfFlatGpu<T, IdxT>::copy()
{
  return std::make_unique<CuvsIvfFlatGpu<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void CuvsIvfFlatGpu<T, IdxT>::search(
  const T* queries, int batch_size, int k, AnnBase::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<AnnBase::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  IdxT* neighbors_IdxT;
  std::optional<rmm::device_uvector<IdxT>> neighbors_storage{std::nullopt};
  if constexpr (sizeof(IdxT) == sizeof(AnnBase::index_type)) {
    neighbors_IdxT = reinterpret_cast<IdxT*>(neighbors);
  } else {
    neighbors_storage.emplace(batch_size * k, raft::resource::get_cuda_stream(handle_));
    neighbors_IdxT = neighbors_storage->data();
  }
  cuvs::neighbors::ivf_flat::search(
    handle_,
    search_params_,
    *index_,
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, index_->dim()),
    raft::make_device_matrix_view<IdxT, int64_t>(neighbors_IdxT, batch_size, k),
    raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k));
  if constexpr (sizeof(IdxT) != sizeof(AnnBase::index_type)) {
    raft::linalg::unaryOp(neighbors,
                          neighbors_IdxT,
                          batch_size * k,
                          raft::cast_op<AnnBase::index_type>(),
                          raft::resource::get_cuda_stream(handle_));
  }
}
}  // namespace cuvs::bench::ann
