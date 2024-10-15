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
#include "../common/thread_pool.hpp"

#include <limits>
#include <raft/core/host_mdspan.hpp>

#include <disk_utils.h>
#include <index.h>
#include <linux_aligned_file_reader.h>
#include <omp.h>
#include <pq_flash_index.h>
#include <raft/util/cudart_utils.hpp>
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
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    uint32_t L_search;
    uint32_t num_threads = omp_get_num_procs();
    // Mode metric_objective;
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
  std::shared_ptr<diskann::IndexWriteParameters> diskann_index_write_params_{nullptr};
  uint32_t max_points_;
  uint32_t build_pq_bytes_ = 0;
  int num_threads_;
  uint32_t L_search_;
  Mode bench_mode_;
  int num_search_threads_;
  std::shared_ptr<fixed_thread_pool> thread_pool_;
  std::string index_path_prefix_;
  std::shared_ptr<diskann::Index<T>> mem_index_{nullptr};
  void initialize_index_(size_t max_points);
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
      .with_num_threads(param.num_threads)
      .build());
}

template <typename T>
void diskann_memory<T>::initialize_index_(size_t max_points)
{
  this->mem_index_ = std::make_shared<diskann::Index<T>>(parse_metric_to_diskann(this->metric_),
                                                         this->dim_,
                                                         max_points,
                                                         diskann_index_write_params_,
                                                         nullptr,
                                                         0,
                                                         false,
                                                         false,
                                                         false,
                                                         build_pq_bytes_ > 0,
                                                         build_pq_bytes_,
                                                         false,
                                                         false);
}
template <typename T>
void diskann_memory<T>::build(const T* dataset, size_t nrow)
{
  initialize_index_(nrow);
  mem_index_->build(dataset, nrow, std::vector<uint32_t>());
}

template <typename T>
void diskann_memory<T>::set_search_param(const search_param_base& param_)
{
  auto param          = dynamic_cast<const search_param&>(param_);
  L_search_           = param.L_search;
  num_search_threads_ = param.num_threads;

  // only latency mode supported with thread pool
  bench_mode_ = Mode::kLatency;

  // Create a pool if multiple query threads have been set and the pool hasn't been created already
  initialize_index_(0);
  this->mem_index_->load(index_path_prefix_.c_str(), num_search_threads_, L_search_);
  bool create_pool = (bench_mode_ == Mode::kLatency && num_search_threads_ > 1 && !thread_pool_);
  if (create_pool) { thread_pool_ = std::make_shared<fixed_thread_pool>(num_search_threads_); }
}

template <typename T>
void diskann_memory<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* indices, float* distances) const
{
  auto f = [&](int i) {
    // diskann in-memory index can only handle a single vector at a time.
    mem_index_->search(queries + i * this->dim_,
                       static_cast<size_t>(k),
                       L_search_,
                       reinterpret_cast<uint64_t*>(indices + i * k),
                       distances + i * k);
  };
  if (bench_mode_ == Mode::kLatency && num_search_threads_ > 1) {
    thread_pool_->submit(f, batch_size);
  } else {
    for (int i = 0; i < batch_size; i++) {
      f(i);
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
  // only save the index path prefix here
  index_path_prefix_ = path_to_index;
}

template <typename T>
class diskann_ssd : public algo<T> {
 public:
  struct build_param {
    uint32_t R;
    uint32_t L_build;
    uint32_t build_pq_bytes = 0;
    float alpha             = 1.2;
    int num_threads         = omp_get_num_procs();
    uint32_t QD             = 192;
  };
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {
    uint32_t L_search;
    uint32_t num_threads        = omp_get_num_procs();
    uint32_t num_nodes_to_cache = 10000;
    int beam_width              = 2;
    // Mode metric_objective;
  };

  diskann_ssd(Metric metric, int dim, const build_param& param);

  void build_from_bin(std::string dataset_path, std::string path_to_index, size_t nrow) override;
  void build(const T* dataset, size_t nrow) override {
    // do nothing. will not be used.
  };

  void set_search_param(const search_param_base& param) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;
  diskann_ssd(const diskann_ssd<T>& other) = default;
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<diskann_ssd<T>>(*this); }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 private:
  std::string index_build_params_str;
  std::shared_ptr<diskann::PQFlashIndex<T, uint32_t>> p_flash_index_;
  int beam_width_;
  uint32_t num_nodes_to_cache_;

  // in-memory index params
  uint32_t build_pq_bytes_ = 0;
  uint32_t max_points_;
  int num_threads_;
  int num_search_threads_;
  uint32_t L_search_;
  Mode bench_mode_;
  std::shared_ptr<fixed_thread_pool> thread_pool_;
  std::string index_path_prefix_;
  std::shared_ptr<AlignedFileReader> reader = nullptr;
};

template <typename T>
diskann_ssd<T>::diskann_ssd(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  // Currently set the indexing RAM budget and the search RAM budget to max value avoid sharding
  uint32_t build_dram_budget  = std::numeric_limits<uint32_t>::max();
  uint32_t search_dram_budget = std::numeric_limits<uint32_t>::max();
  index_build_params_str =
    std::string(std::to_string(param.R)) + " " + std::string(std::to_string(param.L_build)) + " " +
    std::string(std::to_string(search_dram_budget)) + " " +
    std::string(std::to_string(build_dram_budget)) + " " +
    std::string(std::to_string(param.num_threads)) + " " + std::string(std::to_string(false)) +
    " " + std::string(std::to_string(false)) + " " + std::string(std::to_string(0)) + " " +
    std::string(std::to_string(param.QD));
}

template <typename T>
void diskann_ssd<T>::build_from_bin(std::string dataset_path,
                                    std::string path_to_index,
                                    size_t nrow)
{
  diskann::build_disk_index<float>(dataset_path.c_str(),
                                   path_to_index.c_str(),
                                   index_build_params_str.c_str(),
                                   parse_metric_to_diskann(this->metric_),
                                   false,
                                   std::string(""),
                                   false,
                                   std::string(""),
                                   std::string(""),
                                   static_cast<const uint32_t>(0),
                                   static_cast<const uint32_t>(0));
}

template <typename T>
void diskann_ssd<T>::set_search_param(const search_param_base& param_)
{
  auto param          = dynamic_cast<const search_param&>(param_);
  L_search_           = param.L_search;
  num_search_threads_ = param.num_threads;
  num_nodes_to_cache_ = param.num_nodes_to_cache;
  beam_width_         = param.beam_width;

  // only latency mode supported with thread pool
  bench_mode_ = Mode::kLatency;

  bool create_pool = (bench_mode_ == Mode::kLatency && num_search_threads_ > 1 && !thread_pool_);
  if (create_pool) { thread_pool_ = std::make_shared<fixed_thread_pool>(num_search_threads_); }
}

template <typename T>
void diskann_ssd<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto f = [&](int i) {
    // diskann ssd index can only handle a single vector at a time.
    p_flash_index_->cached_beam_search(queries + (i * this->dim_),
                                       static_cast<size_t>(k),
                                       L_search_,
                                       reinterpret_cast<uint64_t*>(neighbors + i * k),
                                       distances + i * k,
                                       beam_width_,
                                       false,
                                       nullptr);
  };
  if (this->bench_mode_ == Mode::kLatency && this->num_search_threads_ > 1) {
    this->thread_pool_->submit(f, batch_size);
  } else {
    for (int i = 0; i < batch_size; i++) {
      f(i);
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
  reader.reset(new LinuxAlignedFileReader());
  p_flash_index_ =
    std::make_shared<diskann::PQFlashIndex<T>>(reader, parse_metric_to_diskann(this->metric_));
  int result = p_flash_index_->load(num_search_threads_, path_to_index.c_str());
  std::vector<uint32_t> node_list;
  p_flash_index_->cache_bfs_levels(num_nodes_to_cache_, node_list);
  p_flash_index_->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();
}
};  // namespace cuvs::bench
