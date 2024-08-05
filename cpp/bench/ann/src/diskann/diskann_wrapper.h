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
#include "cuvs/neighbors/nn_descent.hpp"
#include "linux_aligned_file_reader.h"

#include <cuvs/neighbors/cagra.hpp>
#include <limits>
#include <raft/core/host_mdspan.hpp>

#include <index.h>
#include <omp.h>
#include <pq_flash_index.h>
#include <disk_utils.h>
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
    uint32_t build_pq_bytes = 0;
    float alpha             = 1.2;
    int num_threads         = omp_get_num_procs();
    bool use_cagra_graph;
    uint32_t cagra_graph_degree;
    uint32_t cagra_intermediate_graph_degree;
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    uint32_t L_search;
    uint32_t num_threads = omp_get_num_procs();
    Mode metric_objective;
  };

  diskann_memory(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* indices,
              float* distances) const override;

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
  std::shared_ptr<diskann::IndexWriteParameters> diskann_index_write_params_{nullptr};
  uint32_t max_points_;
  uint32_t build_pq_bytes_;
  std::shared_ptr<diskann::Index<T>> mem_index_{nullptr};
  std::shared_ptr<cuvs::neighbors::cagra::index_params> cagra_index_params_{nullptr};
  int num_threads_;

  uint32_t L_search_;
  Mode metric_objective_;
  int num_search_threads_;
};

template <typename T>
diskann_memory<T>::diskann_memory(Metric metric, int dim, const build_param& param)
  : algo<T>(metric, dim)
{
  assert(this->dim_ > 0);
  num_threads_                = param.num_threads;
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
    auto nn_descent_params =
      cuvs::neighbors::nn_descent::index_params(cagra_index_params.intermediate_graph_degree);
    cagra_index_params.graph_build_params = nn_descent_params;
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
                                                         !use_cagra_graph_ && build_pq_bytes_ > 0,
                                                         use_cagra_graph_ ? 0 : build_pq_bytes_,
                                                         false,
                                                         false,
                                                         use_cagra_graph_,
                                                         cagra_index_params_);
  mem_index_->build(dataset, nrow, std::vector<uint32_t>());
}

template <typename T>
void diskann_memory<T>::set_search_param(const search_param_base& param_)
{
  auto param          = dynamic_cast<const search_param&>(param_);
  this->L_search_     = param.L_search;
  metric_objective_   = param.metric_objective;
  num_search_threads_ = param.num_threads;
}

template <typename T>
void diskann_memory<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* indices, float* distances) const
{
  if (this->metric_objective_ == Mode::kLatency) {
    omp_set_num_threads(num_search_threads_);
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      // mem_index_->search(queries + i * this->dim_,
      //                    static_cast<size_t>(k),
      //                    L_search_,
      //                    indices + i * k,
      //                    distances + i * k);
    }
  } else {
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      // mem_index_->search(queries + i * this->dim_,
      //                    static_cast<size_t>(k),
      //                    L_search_,
      //                    indices + i * k,
      //                    distances + i * k);
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

template <typename T>
class diskann_ssd : public algo<T> {
 public:
  struct build_param {
    uint32_t R;
    uint32_t L_build;
    uint32_t num_threads;
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    uint32_t L_search;
    uint32_t num_threads = omp_get_num_procs();
    Mode metric_objective;
  };

  diskann_ssd(Metric metric, int dim, const build_param& param);

  void build_from_bin(std::string dataset_path, std::string path_to_index, size_t nrow) override;
  void build(const T* dataset, size_t nrow) override {};

  void set_search_param(const search_param_base& param_) override;

  void search(
    const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const override;

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

 protected:
  uint32_t build_pq_bytes_ = 0;
  std::string index_build_params_str;
std::shared_ptr<diskann::PQFlashIndex<T, uint32_t>> p_flash_index_;

  // std::shared_ptr<FixedThreadPool> thread_pool_;
  uint32_t num_nodes_to_cache_;
  uint32_t L_search_;


  bool use_cagra_graph_;
  uint32_t max_points_;
  std::shared_ptr<diskann::Index<T>> mem_index_{nullptr};
  std::shared_ptr<cuvs::neighbors::cagra::index_params> cagra_index_params_{nullptr};

  // uint32_t L_search_;
  Mode metric_objective_;
  // int num_search_threads_;
};

template <typename T>
diskann_ssd<T>::diskann_ssd(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  // Currently set the indexing RAM budget and the search RAM budget to avoid sharding
  uint32_t build_dram_budget  = std::numeric_limits<uint32_t>::max();
  uint32_t search_dram_budget = std::numeric_limits<uint32_t>::max();
  index_build_params_str             = std::string(std::to_string(param.R)) + " " +
                    std::string(std::to_string(param.L_build)) + " " +
                    std::string(std::to_string(search_dram_budget)) + " " +
                    std::string(std::to_string(build_dram_budget)) + " " +
                    std::string(std::to_string(param.num_threads)) + " " +
                    std::string(std::to_string(false)) + " " + std::string(std::to_string(false)) +
                    " " + std::string(std::to_string(0)) + " " + std::string(std::to_string(192));
}

template <typename T>
void diskann_ssd<T>::build_from_bin(std::string dataset_path, std::string path_to_index, size_t nrow)
{
  diskann::build_disk_index<float>(dataset_path.c_str(), path_to_index.c_str(),
                            index_build_params_str.c_str(),
                            parse_metric_to_diskann(this->metric_),
                            false,
                            std::string(""),
                            false,
                            std::string(""),
                            std::string(""),
                            static_cast<const uint32_t>(0),
                            static_cast<const uint32_t>(0),
                            cagra_index_params_);
}

template <typename T>
void diskann_ssd<T>::set_search_param(const search_param_base& param_)
{
  auto param        = dynamic_cast<const search_param&>(param_);
  this->L_search_   = param.L_search;
  // metric_objective_ = param.metric_objective;
}

template <typename T>
void diskann_ssd<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  std::vector<uint32_t> node_list;
  p_flash_index_->cache_bfs_levels(num_nodes_to_cache_, node_list);
  p_flash_index_->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();

  if (this->metric_objective_ == Mode::kLatency) {
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      p_flash_index_->cached_beam_search(queries + (i * this->dim_),
                                         static_cast<size_t>(k),
                                         L_search_,
                                         reinterpret_cast<size_t*>(neighbors + i * k),
                                         distances + i * k,
                                         2);
    }
  } else {
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
      p_flash_index_->cached_beam_search(queries + (i * this->dim_),
                                         static_cast<size_t>(k),
                                         L_search_,
                                         reinterpret_cast<size_t*>(neighbors + i * k),
                                         distances + i * k,
                                         2);
    }
  }
}

template <typename T>
void diskann_ssd<T>::save(const std::string& path_to_index) const
{
  // Nothing to do here. Index already saved in build stage.
}

template <typename T>
void diskann_ssd<T>::load(const std::string& path_to_index)
{
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  int result = p_flash_index_->load(omp_get_num_procs(), path_to_index.c_str());
}
};  // namespace cuvs::bench
