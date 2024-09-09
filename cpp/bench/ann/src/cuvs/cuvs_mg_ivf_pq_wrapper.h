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

#include "cuvs_ann_bench_utils.h"
#include "cuvs_ivf_pq_wrapper.h"
#include <cuvs/neighbors/mg.hpp>

namespace cuvs::bench {
using namespace cuvs::neighbors;

template <typename T, typename IdxT>
class cuvs_mg_ivf_pq : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;

  using build_param = cuvs::neighbors::mg::index_params<ivf_pq::index_params>;

  struct search_param : public cuvs::bench::cuvs_ivf_pq<T, IdxT>::search_param {
    cuvs::neighbors::mg::sharded_merge_mode merge_mode;
  };

  cuvs_mg_ivf_pq(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), index_params_(param)
  {
    index_params_.metric = parse_metric_type(metric);
    clique_              = std::make_shared<cuvs::neighbors::mg::nccl_clique>();
  }

  void build(const T* dataset, size_t nrow) final;
  void set_search_param(const search_param_base& param) override;
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    const auto& handle = clique_->set_current_device_to_root_rank();
    auto stream        = raft::resource::get_cuda_stream(handle);
    return stream;
  }

  [[nodiscard]] auto uses_stream() const noexcept -> bool override { return false; }

  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override;

 private:
  std::shared_ptr<cuvs::neighbors::mg::nccl_clique> clique_;
  build_param index_params_;
  cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params_;
  std::shared_ptr<cuvs::neighbors::mg::index<cuvs::neighbors::ivf_pq::index<IdxT>, T, IdxT>> index_;
};

template <typename T, typename IdxT>
void cuvs_mg_ivf_pq<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_view =
    raft::make_host_matrix_view<const T, int64_t, raft::row_major>(dataset, IdxT(nrow), IdxT(dim_));
  const auto& handle = clique_->set_current_device_to_root_rank();
  auto idx           = cuvs::neighbors::mg::build(handle, *clique_, index_params_, dataset_view);
  index_ =
    std::make_shared<cuvs::neighbors::mg::index<cuvs::neighbors::ivf_pq::index<IdxT>, T, IdxT>>(
      std::move(idx));
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_pq<T, IdxT>::set_search_param(const search_param_base& param)
{
  auto sp = dynamic_cast<const search_param&>(param);
  // search_params_ = static_cast<mg::search_params<ivf_pq::search_params>>(sp.pq_param);
  ivf_pq::search_params* search_params_ptr_ = static_cast<ivf_pq::search_params*>(&search_params_);
  *search_params_ptr_                       = sp.pq_param;
  search_params_.merge_mode                 = sp.merge_mode;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_pq<T, IdxT>::save(const std::string& file) const
{
  const auto& handle = clique_->set_current_device_to_root_rank();
  cuvs::neighbors::mg::serialize(handle, *clique_, *index_, file);
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_pq<T, IdxT>::load(const std::string& file)
{
  const auto& handle = clique_->set_current_device_to_root_rank();
  index_ =
    std::make_shared<cuvs::neighbors::mg::index<cuvs::neighbors::ivf_pq::index<IdxT>, T, IdxT>>(
      std::move(cuvs::neighbors::mg::deserialize_pq<T, IdxT>(handle, *clique_, file)));
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_mg_ivf_pq<T, IdxT>::copy()
{
  return std::make_unique<cuvs_mg_ivf_pq<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_mg_ivf_pq<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto queries_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
    queries, IdxT(batch_size), IdxT(dim_));
  auto neighbors_view = raft::make_host_matrix_view<IdxT, int64_t, raft::row_major>(
    (IdxT*)neighbors, IdxT(batch_size), IdxT(k));
  auto distances_view = raft::make_host_matrix_view<float, int64_t, raft::row_major>(
    distances, IdxT(batch_size), IdxT(k));

  const auto& handle = clique_->set_current_device_to_root_rank();
  cuvs::neighbors::mg::search(handle,
                              *clique_,
                              *index_,
                              search_params_,
                              queries_view,
                              neighbors_view,
                              distances_view,
                              search_params_.merge_mode);
}

}  // namespace cuvs::bench