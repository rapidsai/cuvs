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
#include "cuvs_ann_bench_utils.h"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/host_mdspan.hpp>

#include <index.h>
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
class diskann_memory : public algo<T>, public algo_gpu {
 public:
  struct build_param {
    uint32_t R;
    int num_threads = omp_get_num_procs();
    bool use_cuvs_cagra_graph;
    uint32_t cagra_graph_degree;
    uint32_t cagra_intermediate_graph_degree;
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    uint32_t L_search;
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
  bool use_pq_build_       = false;
  uint32_t build_pq_bytes_ = 0;
  std::shared_ptr<diskann::Index<T>> diskann_index_{nullptr};
  uint32_t L_search_;
  uint32_t cagra_graph_degree_              = 64;
  uint32_t cagra_intermediate_graph_degree_ = 128;
  uint32_t max_points_;
  // std::shared_ptr<FixedThreadPool> thread_pool_;
  Mode metric_objective_;
};

template <typename T>
diskann_memory<T>::diskann_memory(Metric metric, int dim, const build_param& param)
  : algo<T>(metric, dim)
{
  assert(this->dim_ > 0);
  auto diskann_index_write_params = std::make_shared<diskann::IndexWriteParameters>(
    diskann::IndexWriteParametersBuilder(param.L_build, param.R)
      .with_filter_list_size(0)
      .with_alpha(param.alpha)
      .with_saturate_graph(false)
      .with_num_threads(param.num_threads)
      .build());
  use_cagra_graph_                 = param.use_cagra_graph;
  build_pq_bytes_                  = 0;
  cagra_graph_degree_              = param.cagra_graph_degree;
  cagra_intermediate_graph_degree_ = param.cagra_intermediate_graph_degree;
  cuvs::neighbors::cagra::index_params cuvs_cagra_index_params;
  cuvs_cagra_index_params.intermediate_graph_degree = param.cagra_intermediate_graph_degree;
  cuvs_cagra_index_params.graph_degree              = param.cagra_graph_degree;

  this->diskann_index_ = std::make_shared<diskann::Index<T>>(parse_metric_to_diskann(metric),
                                                             dim,
                                                             10000000,
                                                             diskann_index_write_params,
                                                             nullptr,
                                                             0,
                                                             false,
                                                             false,
                                                             false,
                                                             false,
                                                             this->build_pq_bytes_,
                                                             false,
                                                             false,
                                                             param.use_cagra_graph,
                                                             param.cagra_graph_degree);
}

template <typename T>
void diskann_memory<T>::build(const T* dataset, size_t nrow)
{
  max_points_ = nrow;
  // std::cout << "num_threads" << this->diskann_index_write_params_->num_threads << std::endl;

  if (use_cagra_graph_) {
    std::optional<raft::host_matrix<uint32_t, int64_t>> intermediate_graph(
      raft::make_host_matrix<uint32_t, int64_t>(nrow, cagra_intermediate_graph_degree_));

    std::vector<uint32_t> knn_graph(nrow * cagra_graph_degree_);
    auto knn_graph_view =
      raft::make_host_matrix_view<uint32_t, int64_t>(knn_graph.data(), nrow, cagra_graph_degree_);
    auto dataset_view = raft::make_host_matrix_view<const T, int64_t>(
      dataset, static_cast<int64_t>(nrow), (int64_t)this->dim_);
    raft::resources res;
    auto start                     = std::chrono::high_resolution_clock::now();
    auto nn_descent_params         = raft::neighbors::experimental::nn_descent::index_params();
    nn_descent_params.graph_degree = cagra_intermediate_graph_degree_;
    nn_descent_params.intermediate_graph_degree = 1.5 * cagra_intermediate_graph_degree_;
    nn_descent_params.max_iterations            = 20;
    // auto ivf_pq_params         =
    // raft::neighbors::ivf_pq::index_params::from_dataset(dataset_view); ivf_pq_params.n_lists =
    // static_cast<uint32_t>(nrow / 2500);

    raft::neighbors::cagra::build_knn_graph(
      res, dataset_view, intermediate_graph->view(), nn_descent_params);
    raft::neighbors::cagra::optimize(res, intermediate_graph->view(), knn_graph_view);
    // free intermediate graph before trying to create the index
    intermediate_graph.reset();

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "cagra graph built in" << duration << " seconds" << std::endl;
    diskann_index_->build(dataset, nrow, std::vector<uint32_t>(), knn_graph);
  } else {
    diskann_index_->build(dataset, nrow, std::vector<uint32_t>());
  }
}

template <typename T>
void diskann_memory<T>::set_search_param(const search_param_base& param_)
{
  auto param        = dynamic_cast<const search_param&>(param_);
  this->L_search_   = param.L_search;
  metric_objective_ = param.metric_objective;
}

template <typename T>
void diskann_memory<T>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
{
  // std::cout << "num_search_threads" << diskann_index_write_params_->num_threads << std::endl;
  if (this->metric_objective_ == Mode::kLatency) {
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      diskann_index_->search(queries + i * this->dim_,
                             static_cast<size_t>(k),
                             L_search_,
                             neighbors + i * k,
                             distances + i * k);
    }
  } else {
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      diskann_index_->search(queries + i * this->dim_,
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
  this->diskann_index_->save(path_to_index.c_str());
}

template <typename T>
void diskann_memory<T>::load(const std::string& path_to_index)
{
  diskann_index_->load(path_to_index.c_str(), 80, 100);
}

/*******************************************************************
 */

template <typename T>
class diskann_ssd : public algo<T> {
 public:
  struct build_param {
    uint32_t R;
    uint32_t L_build;
    float alpha;
    int num_threads = omp_get_num_procs();
    bool use_cagra_graph;
    bool filtered_index;
    uint32_t cagra_graph_degree;
    uint32_t cagra_intermediate_graph_degree;
  };

  using typename algo<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    uint32_t L_search;
  };

  diskann_ssd(Metric metric, int dim, const build_param& param);

  void build(const char* dataset_path, size_t nrow) override;

  void set_search_param(const AnnSearchParam& param) override;
  void search(
    const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;
  diskann_ssd(const diskann_ssd<T>& other) = default;
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<diskann_ssd<T>>(*this); }

  algo_property get_preference() const override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 private:
  uint32_t build_pq_bytes_ = 0;
  bool use_cagra_graph_;
  std::shared_ptr<diskann::IndexSearchParams> diskann_index_write_params_{nullptr};
  uint32_t max_points_;
  std::shared_ptr<diskann::Index<T>> mem_index_{nullptr};
  std::unique_ptr<diskann::PQFlashIndex<T, uint32_t>> p_flash_index_;
  uint32_t cagra_graph_degree_;
  uint32_t cagra_intermediate_graph_degree_;

  // std::shared_ptr<FixedThreadPool> thread_pool_;
  uint32_t L_search_;
  Mode metric_objective_;
  uint32_t num_nodes_to_cache_;
};

template <typename T>
void diskann_ssd<T>::build(std::string dataset_file, size_t nrow)
{
  std::shared_ptr<cuvs::neighbors::cagra::index_params> cagra_index_params_ptr{nullptr};
  if (use_cagra_graph_) {
    cuvs::neighbors::cagra::index_params cagra_index_params;
    cagra_index_params.graph_degree              = cagra_graph_degree_;
    cagra_index_params.intermediate_graph_degree = cagra_intermediate_graph_degree_;
    auto ivf_pq_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
      raft::matrix_extent<int64_t>(nrow, this->dim_), parse_metric_type(this->metric_));
    if (build_pq_bytes_ > 0) ivf_pq_params.build_params.pq_dim = build_pq_bytes_;
    ivf_pq_params.build_params.pq_bits = 8;
    cagra_index_params.graph_build_params = ivf_pq_params;
    cagra_index_params_ptr.reset(&cagra_index_params);
  }
  this->mem_index_ = std::make_shared<diskann::Index<T>>(parse_metric_to_diskann(this->metric_),
                                                             this->dim_,
                                                             nrow,
                                                             diskann_index_write_params_,
                                                             nullptr,
                                                             0,
                                                             false,
                                                             false,
                                                             false,
                                                             build_pq_bytes_ > 0,
                                                             this->build_pq_bytes_,
                                                             false,
                                                             false,
                                                             use_cagra_graph_,
                                                             cagra_index_params_ptr);
  this->mem_index_->build(dataset_file.c_str(), nrow);
}

template <typename T>
void diskann_ssd<T>::set_search_param(const AnnSearchParam& param_)
{
  auto param        = dynamic_cast<const SearchParam&>(param_);
  this->L_search_   = param.L_search;
  metric_objective_ = param.metric_objective;
}

template <typename T>
void diskann_ssd<T>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
{
  std::vector<uint32_t> node_list;
  p_flash_index_->cache_bfs_levels(num_nodes_to_cache_, node_list);
  p_flash_index_->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();

  if (this->metric_objective_ == Objective::LATENCY) {
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      p_flash_index_->cached_beam_search(queries + (i * this->dim_),
                                       static_cast<size_t>(k),
                                       L_search_,
                                       neighbors + i * k,
                                       distances + i * k,
                                       2);
    }
  } else {
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      p_flash_index_->cached_beam_search(queries + (i * this->dim_),
                                       static_cast<size_t>(k),
                                       L_search_,
                                       neighbors + i * k,
                                       distances + i * k,
                                       2);
    }
  }
}

template <typename T>
void diskann_ssd<T>::save(const std::string& path_to_index) const
{
  this->diskann_index_->save(path_to_index.c_str());
}

template <typename T>
void diskann_ssd<T>::load(const std::string& path_to_index)
{
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  int result = p_flash_index_->load(omp_get_num_procs(), path_to_index.c_str());
}

};  // namespace cuvs::bench
