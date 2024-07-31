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

#include "../common/ann_types.hpp"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/host_mdspan.hpp>

#include <index.h>
#include <pq_flash_index.h>
#include <omp.h>
#include <utils.h>

#include <chrono>
#include <memory>
#include <vector>

namespace cuvs::bench {

diskann::Metric parse_metric_to_diskann(cuvs::bench::Metric metric)
{
  if (metric == cuvs::bench::Metric::kInnerProduct) {
    return diskann::Metric::INNER_PRODUCT;
  } else if (metric == cuvs::bench::Metric::kEuclidean) {
    return diskann::Metric::L2;
  } else {
    throw std::runtime_error("currently only inner product and L2 supported for benchmarking");
  }
}

template <typename T>
class diskann_memory : public algo<T> {
 public:
  struct build_param {
    uint32_t R;
    uint32_t L_build;
    float alpha = 1.2;
    int num_threads = omp_get_num_procs();
    bool use_cagra_graph;
    uint32_t cagra_graph_degree;
    uint32_t cagra_intermediate_graph_degree;
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    uint32_t L_search;
    uint32_t num_threads = omp_get_num_procs();
  };

  diskann_memory(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param) override;
  void search(
    const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;
  diskann_memory(const diskann_memory<T>& other) = default;
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<diskann_memory<T>>(*this); }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 private:
  bool use_cagra_graph_;
  std::shared_ptr<diskann::IndexSearchParams> diskann_index_write_params_{nullptr};
  uint32_t max_points_;
  std::shared_ptr<diskann::Index<T>> mem_index_{nullptr};
  std::shared_ptr<cuvs::neighbors::cagra::index_params> cagra_index_params_{nullptr};
  int num_threads_;

  // std::shared_ptr<FixedThreadPool> thread_pool_;
  uint32_t L_search_;
  Mode metric_objective_;
  int num_search_threads_;
};

template <typename T>
diskann_memory<T>::diskann_memory(Metric metric, int dim, const build_param& param)
  : algo<T>(metric, dim)
{
  assert(this->dim_ > 0);
  num_threads_ = param.num_threads;
  diskann_index_write_params_ = std::make_shared<diskann::IndexWriteParameters>(
    diskann::IndexWriteParametersBuilder(param.L_build, param.R)
      .with_filter_list_size(0)
      .with_alpha(param.alpha)
      .with_saturate_graph(false)
      .with_num_threads(num_threads_)
      .build());
  use_cagra_graph_ = param.use_cagra_graph;
  if (use_cagra_graph_) {
    cuvs::neighbors::cagra::index_params cagra_index_params;
    cagra_index_params.intermediate_graph_degree = param.cagra_intermediate_graph_degree;
    cagra_index_params.graph_degree              = param.cagra_graph_degree;
    cagra_index_params.graph_build_params.nn_descent_params.graph_degree =
      cagra_index_params.intermediate_graph_degree;
    cagra_index_params.graph_build_params.nn_descent_params.intermediate_graph_degree =
      1.5 * cagra_index_params.intermediate_graph_degree;
    cagra_index_params_ =
      std::make_shared<cuvs::neighbors::cagra::index_params>(cagra_index_params);
  }
}

template <typename T>
void diskann_memory<T>::build(const T* dataset, size_t nrow)
{
  max_points_      = nrow;
  this->mem_index_ = std::make_shared<diskann::Index<T>>(parse_metric_to_diskann(this->metric_),
                                                         this->dim_,
                                                         max_points_,
                                                         diskann_index_write_params_,
                                                         nullptr,
                                                         0,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         0,
                                                         false,
                                                         false,
                                                         use_cagra_graph_,
                                                         cagra_index_params_);
}

template <typename T>
void diskann_memory<T>::set_search_param(const search_param_base& param_)
{
  auto param        = dynamic_cast<const search_param&>(param_);
  this->L_search_   = param.L_search;
  metric_objective_ = param.metric_objective;
  num_search_threads_ = param.num_threads;
}

template <typename T>
void diskann_memory<T>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
{
  if (this->metric_objective_ == Mode::kLatency) {
    omp_set_num_threads(num_search_threads_);
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      mem_index_->search(queries + i * this->dim_,
                         static_cast<size_t>(k),
                         L_search_,
                         neighbors + i * k,
                         distances + i * k);
    }
  } else {
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      mem_index_->search(queries + i * this->dim_,
                         static_cast<size_t>(k),
                         L_search_,
                         neighbors + i * k,
                         distances + i * k);
    }
  }
}

template <typename T>
void diskann_memory<T>::save(const std::string& path_to_index) const
{
  this->mem_index_->save(path_to_index.c_str());
}

template <typename T>
void diskann_memory<T>::load(const std::string& path_to_index)
{
  this->mem_index_->load(path_to_index.c_str(), num_threads_, 100);
}

// template <typename T>
// class diskann_ssd : public diskann_memory<T> {
//  public:
//   struct build_param : public diskann_memory<T>::build_param {
//     uint32_t build_pq_bytes;
//   };

//   diskann_ssd(Metric metric, int dim, const build_param& param);

//   void build(const char* dataset_path, size_t nrow);

//   void search(
//     const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const override;

//   void save(const std::string& path_to_index) const override;
//   void load(const std::string& path_to_index) override;
//   diskann_ssd(const diskann_ssd<T>& other) = default;
//   std::unique_ptr<algo<T>> copy() override { return std::make_unique<diskann_ssd<T>>(*this); }

//   algo_property get_preference() const override
//   {
//     algo_property property;
//     property.dataset_memory_type = MemoryType::kHost;
//     property.query_memory_type   = MemoryType::kHost;
//     return property;
//   }

//  protected:
//   uint32_t build_pq_bytes_ = 0;
//   std::unique_ptr<diskann::PQFlashIndex<T, uint32_t>> p_flash_index_;

//   // std::shared_ptr<FixedThreadPool> thread_pool_;
//   uint32_t num_nodes_to_cache_;
// };

// template <typename T>
// diskann_ssd<T>::diskann_ssd(Metric metric, int dim, const build_param& param)
//   : algo<T>(metric, dim)
// {
//   assert(this->dim_ > 0);
//   this->num_threads_ = param.num_threads;
//   diskann_index_write_params_ = std::make_shared<diskann::IndexWriteParameters>(
//     diskann::IndexWriteParametersBuilder(param.L_build, param.R)
//       .with_filter_list_size(0)
//       .with_alpha(param.alpha)
//       .with_saturate_graph(false)
//       .with_num_threads(num_threads_)
//       .build());
//   use_cagra_graph_ = param.use_cagra_graph;
//   if (use_cagra_graph_) {
//     cuvs::neighbors::cagra::index_params cagra_index_params;
//     cagra_index_params.intermediate_graph_degree = param.cagra_intermediate_graph_degree;
//     cagra_index_params.graph_degree              = param.cagra_graph_degree;
//     cagra_index_params.graph_build_params.nn_descent_params.graph_degree =
//       cagra_index_params.intermediate_graph_degree;
//     cagra_index_params.graph_build_params.nn_descent_params.intermediate_graph_degree =
//       1.5 * cagra_index_params.intermediate_graph_degree;
//     cagra_index_params_ =
//       std::make_shared<cuvs::neighbors::cagra::index_params>(cagra_index_params);
//   }
// }

// template <typename T>
// void diskann_ssd<T>::build(std::string dataset_path, size_t nrow)
// {
//   std::shared_ptr<cuvs::neighbors::cagra::index_params> cagra_index_params_ptr{nullptr};
//   if (use_cagra_graph_) {
//     cuvs::neighbors::cagra::index_params cagra_index_params;
//     cagra_index_params.graph_degree              = cagra_graph_degree_;
//     cagra_index_params.intermediate_graph_degree = cagra_intermediate_graph_degree_;
//     auto ivf_pq_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
//       raft::matrix_extent<int64_t>(nrow, this->dim_), parse_metric_type(this->metric_));
//     if (build_pq_bytes_ > 0) ivf_pq_params.build_params.pq_dim = build_pq_bytes_;
//     ivf_pq_params.build_params.pq_bits    = 8;
//     cagra_index_params.graph_build_params = ivf_pq_params;
//     cagra_index_params_ptr.reset(&cagra_index_params);
//   }
//   this->mem_index_ = std::make_shared<diskann::Index<T>>(parse_metric_to_diskann(this->metric_),
//                                                          this->dim_,
//                                                          nrow,
//                                                          diskann_index_write_params_,
//                                                          nullptr,
//                                                          0,
//                                                          false,
//                                                          false,
//                                                          false,
//                                                          build_pq_bytes_ > 0,
//                                                          this->build_pq_bytes_,
//                                                          false,
//                                                          false,
//                                                          use_cagra_graph_,
//                                                          cagra_index_params_ptr);
//   this->mem_index_->build(dataset_file.c_str(), nrow);
// }

// template <typename T>
// void diskann_ssd<T>::set_search_param(const AnnSearchParam& param_)
// {
//   auto param        = dynamic_cast<const SearchParam&>(param_);
//   this->L_search_   = param.L_search;
//   metric_objective_ = param.metric_objective;
// }

// template <typename T>
// void diskann_ssd<T>::search(
//   const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
// {
//   std::vector<uint32_t> node_list;
//   p_flash_index_->cache_bfs_levels(num_nodes_to_cache_, node_list);
//   p_flash_index_->load_cache_list(node_list);
//   node_list.clear();
//   node_list.shrink_to_fit();

//   if (this->metric_objective_ == Mode::kLatency) {
//     omp_set_num_threads(num_search_threads_);
// #pragma omp parallel for
//     for (int64_t i = 0; i < (int64_t)batch_size; i++) {
//       p_flash_index_->cached_beam_search(queries + (i * this->dim_),
//                                          static_cast<size_t>(k),
//                                          L_search_,
//                                          neighbors + i * k,
//                                          distances + i * k,
//                                          2);
//     }
//   } else {
//     for (int64_t i = 0; i < (int64_t)batch_size; i++) {
//       p_flash_index_->cached_beam_search(queries + (i * this->dim_),
//                                          static_cast<size_t>(k),
//                                          L_search_,
//                                          neighbors + i * k,
//                                          distances + i * k,
//                                          2);
//     }
//   }
// }

// template <typename T>
// void diskann_ssd<T>::save(const std::string& path_to_index) const
// {
//   this->diskann_index_->save(path_to_index.c_str());
// }

// template <typename T>
// void diskann_ssd<T>::load(const std::string& path_to_index)
// {
//   std::shared_ptr<AlignedFileReader> reader = nullptr;
//   reader.reset(new LinuxAlignedFileReader());
//   int result = p_flash_index_->load(omp_get_num_procs(), path_to_index.c_str());
// }

};  // namespace cuvs::bench
