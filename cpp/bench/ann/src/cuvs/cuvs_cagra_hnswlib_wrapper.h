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

#include "../hnswlib/hnswlib_wrapper.h"
#include "cuvs_cagra_wrapper.h"

#include <memory>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_cagra_hnswlib : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using build_param       = typename cuvs_cagra<T, IdxT>::build_param;
  using search_param      = typename hnsw_lib<T>::search_param;

  cuvs_cagra_hnswlib(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim),
      cagra_build_{metric, dim, param, concurrent_searches},
      // hnsw_lib param values don't matter since we don't build with hnsw_lib
      hnswlib_search_{metric, dim, typename hnsw_lib<T>::build_param{50, 100}}
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
  cuvs_cagra<T, IdxT> cagra_build_;
  hnsw_lib<T> hnswlib_search_;
};

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::build(const T* dataset, size_t nrow)
{
  cagra_build_.build(dataset, nrow);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::set_search_param(const search_param_base& param_)
{
  hnswlib_search_.set_search_param(param_);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::save(const std::string& file) const
{
  cagra_build_.save_to_hnswlib(file);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::load(const std::string& file)
{
  hnswlib_search_.load(file);
  hnswlib_search_.set_base_layer_only();
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  hnswlib_search_.search(queries, batch_size, k, neighbors, distances);
}

}  // namespace cuvs::bench
