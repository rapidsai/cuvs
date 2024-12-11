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

#include "cuvs_cagra_wrapper.h"
#include <cuvs/neighbors/hnsw.hpp>

#include <memory>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_cagra_hnswlib : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct build_param {
    typename cuvs_cagra<T, IdxT>::build_param cagra_build_param;
    cuvs::neighbors::hnsw::index_params hnsw_index_params;
  };

  struct search_param : public search_param_base {
    cuvs::neighbors::hnsw::search_params hnsw_search_param;
  };

  cuvs_cagra_hnswlib(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim),
      build_param_{param},
      cagra_build_{metric, dim, param.cagra_build_param, concurrent_searches}
  {
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return cagra_build_.get_sync_stream();
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override
  {
    return std::make_unique<cuvs_cagra_hnswlib<T, IdxT>>(*this);
  }

 private:
  raft::resources handle_{};
  build_param build_param_;
  search_param search_param_;
  cuvs_cagra<T, IdxT> cagra_build_;
  std::shared_ptr<cuvs::neighbors::hnsw::index<T>> hnsw_index_;
};

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::build(const T* dataset, size_t nrow)
{
  cagra_build_.build(dataset, nrow);
  auto* cagra_index      = cagra_build_.get_index();
  auto host_dataset_view = raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
  auto opt_dataset_view =
    std::optional<raft::host_matrix_view<const T, int64_t>>(std::move(host_dataset_view));
  hnsw_index_ = cuvs::neighbors::hnsw::from_cagra(
    handle_, build_param_.hnsw_index_params, *cagra_index, opt_dataset_view);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::set_search_param(const search_param_base& param_)
{
  search_param_ = dynamic_cast<const search_param&>(param_);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::hnsw::serialize(handle_, file, *(hnsw_index_.get()));
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::load(const std::string& file)
{
  cuvs::neighbors::hnsw::index<T>* idx = nullptr;
  cuvs::neighbors::hnsw::deserialize(handle_,
                                     build_param_.hnsw_index_params,
                                     file,
                                     this->dim_,
                                     parse_metric_type(this->metric_),
                                     &idx);
  hnsw_index_ = std::shared_ptr<cuvs::neighbors::hnsw::index<T>>(idx);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  // Only Latency mode is supported for now
  auto queries_view =
    raft::make_host_matrix_view<const T, int64_t>(queries, batch_size, this->dim_);
  auto neighbors_view = raft::make_host_matrix_view<uint64_t, int64_t>(
    reinterpret_cast<uint64_t*>(neighbors), batch_size, k);
  auto distances_view = raft::make_host_matrix_view<float, int64_t>(distances, batch_size, k);

  cuvs::neighbors::hnsw::search(handle_,
                                search_param_.hnsw_search_param,
                                *(hnsw_index_.get()),
                                queries_view,
                                neighbors_view,
                                distances_view);
}

}  // namespace cuvs::bench
