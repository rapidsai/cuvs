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
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/refine.cuh>
#include <raft/util/cudart_utils.hpp>

#include <type_traits>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_ivf_pq : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;

  struct search_param : public search_param_base {
    cuvs::neighbors::ivf_pq::search_params pq_param;
    float refine_ratio = 1.0f;
    [[nodiscard]] auto needs_dataset() const -> bool override { return refine_ratio > 1.0f; }
  };

  using build_param = cuvs::neighbors::ivf_pq::index_params;

  cuvs_ivf_pq(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric = parse_metric_type(metric);
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param) override;
  void set_search_dataset(const T* dataset, size_t nrow) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  build_param index_params_;
  cuvs::neighbors::ivf_pq::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::ivf_pq::index<IdxT>> index_;
  int dimension_;
  float refine_ratio_ = 1.0;
  raft::device_matrix_view<const T, IdxT> dataset_;
};

template <typename T, typename IdxT>
void cuvs_ivf_pq<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::ivf_pq::serialize(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_ivf_pq<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<cuvs::neighbors::ivf_pq::index<IdxT>>(handle_, index_params_, dim_);
  cuvs::neighbors::ivf_pq::deserialize(handle_, file, index_.get());
}

template <typename T, typename IdxT>
void cuvs_ivf_pq<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_v = raft::make_device_matrix_view<const T, IdxT>(dataset, IdxT(nrow), dim_);
  std::make_shared<cuvs::neighbors::ivf_pq::index<IdxT>>(
    std::move(cuvs::neighbors::ivf_pq::build(handle_, index_params_, dataset_v)))
    .swap(index_);
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_ivf_pq<T, IdxT>::copy()
{
  return std::make_unique<cuvs_ivf_pq<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_ivf_pq<T, IdxT>::set_search_param(const search_param_base& param)
{
  auto sp        = dynamic_cast<const search_param&>(param);
  search_params_ = sp.pq_param;
  refine_ratio_  = sp.refine_ratio;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void cuvs_ivf_pq<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  dataset_ = raft::make_device_matrix_view<const T, IdxT>(dataset, nrow, index_->dim());
}

template <typename T, typename IdxT>
void cuvs_ivf_pq<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  IdxT* neighbors_idx_t;
  std::optional<rmm::device_uvector<IdxT>> neighbors_storage{std::nullopt};
  if constexpr (sizeof(IdxT) == sizeof(algo_base::index_type)) {
    neighbors_idx_t = reinterpret_cast<IdxT*>(neighbors);
  } else {
    neighbors_storage.emplace(batch_size * k, raft::resource::get_cuda_stream(handle_));
    neighbors_idx_t = neighbors_storage->data();
  }

  auto queries_view =
    raft::make_device_matrix_view<const T, uint32_t>(queries, batch_size, dimension_);
  auto neighbors_view =
    raft::make_device_matrix_view<IdxT, uint32_t>(neighbors_idx_t, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, uint32_t>(distances, batch_size, k);

  cuvs::neighbors::ivf_pq::search(
    handle_, search_params_, *index_, queries_view, neighbors_view, distances_view);

  if constexpr (sizeof(IdxT) != sizeof(algo_base::index_type)) {
    raft::linalg::unaryOp(neighbors,
                          neighbors_idx_t,
                          batch_size * k,
                          raft::cast_op<algo_base::index_type>(),
                          raft::resource::get_cuda_stream(handle_));
  }
}

template <typename T, typename IdxT>
void cuvs_ivf_pq<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    auto queries_v = raft::make_device_matrix_view<const T, algo_base::index_type>(
      queries, batch_size, dimension_);
    auto candidate_ixs =
      raft::make_device_matrix<algo_base::index_type, algo_base::index_type>(res, batch_size, k0);
    auto candidate_dists =
      raft::make_device_matrix<float, algo_base::index_type>(res, batch_size, k0);
    search_base(
      queries, batch_size, k0, candidate_ixs.data_handle(), candidate_dists.data_handle());
    refine_helper(
      res, dataset_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }
}
}  // namespace cuvs::bench
