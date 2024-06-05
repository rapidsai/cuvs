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
#include "../common/thread_pool.hpp"
#include "../common/util.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace cuvs::bench {

template <typename T>
struct hnsw_dist_t {
  using type = void;
};

template <>
struct hnsw_dist_t<float> {
  using type = float;
};

template <>
struct hnsw_dist_t<uint8_t> {
  using type = int;
};

template <>
struct hnsw_dist_t<int8_t> {
  using type = int;
};

template <typename T>
class hnsw_lib : public algo<T> {
 public:
  // https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
  struct build_param {
    int m;
    int ef_construction;
    int num_threads = omp_get_num_procs();
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    int ef;
    int num_threads = 1;
  };

  hnsw_lib(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param) override;
  void search(const T* query,
              int batch_size,
              int k,
              algo_base::index_type* indices,
              float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;
  auto copy() -> std::unique_ptr<algo<T>> override { return std::make_unique<hnsw_lib<T>>(*this); };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  void set_base_layer_only() { appr_alg_->base_layer_only = true; }

 private:
  void get_search_knn_results(const T* query,
                              int k,
                              algo_base::index_type* indices,
                              float* distances) const;

  std::shared_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::shared_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;

  using algo<T>::metric_;
  using algo<T>::dim_;
  int ef_construction_;
  int m_;
  int num_threads_;
  std::shared_ptr<fixed_thread_pool> thread_pool_;
  Mode bench_mode_;
};

template <typename T>
hnsw_lib<T>::hnsw_lib(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  assert(dim_ > 0);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t>);
  if constexpr (std::is_same_v<T, uint8_t>) {
    if (metric_ != Metric::kEuclidean) {
      throw std::runtime_error("hnswlib<uint8_t> only supports Euclidean distance");
    }
  }

  ef_construction_ = param.ef_construction;
  m_               = param.m;
  num_threads_     = param.num_threads;
}

template <typename T>
void hnsw_lib<T>::build(const T* dataset, size_t nrow)
{
  if constexpr (std::is_same_v<T, float>) {
    if (metric_ == Metric::kInnerProduct) {
      space_ = std::make_shared<hnswlib::InnerProductSpace>(dim_);
    } else {
      space_ = std::make_shared<hnswlib::L2Space>(dim_);
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    space_ = std::make_shared<hnswlib::L2SpaceI<T>>(dim_);
  }

  appr_alg_ = std::make_shared<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    space_.get(), nrow, m_, ef_construction_);

  thread_pool_                  = std::make_shared<fixed_thread_pool>(num_threads_);
  const size_t items_per_thread = nrow / (num_threads_ + 1);

  thread_pool_->submit(
    [&](size_t i) {
      if (i < items_per_thread && i % 10000 == 0) {
        char buf[20];
        std::time_t now = std::time(nullptr);
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        printf("%s building %zu / %zu\n", buf, i, items_per_thread);
        fflush(stdout);
      }

      appr_alg_->addPoint(dataset + i * dim_, i);
    },
    nrow);
}

template <typename T>
void hnsw_lib<T>::set_search_param(const search_param_base& param_)
{
  auto param     = dynamic_cast<const search_param&>(param_);
  appr_alg_->ef_ = param.ef;
  num_threads_   = param.num_threads;
  // bench_mode_ = param.metric_objective;
  bench_mode_ = Mode::kLatency;  // TODO(achirkin): pass the benchmark mode in the algo parameters

  // Create a pool if multiple query threads have been set and the pool hasn't been created already
  bool create_pool = (bench_mode_ == Mode::kLatency && num_threads_ > 1 && !thread_pool_);
  if (create_pool) { thread_pool_ = std::make_shared<fixed_thread_pool>(num_threads_); }
}

template <typename T>
void hnsw_lib<T>::search(
  const T* query, int batch_size, int k, algo_base::index_type* indices, float* distances) const
{
  auto f = [&](int i) {
    // hnsw can only handle a single vector at a time.
    get_search_knn_results(query + i * dim_, k, indices + i * k, distances + i * k);
  };
  if (bench_mode_ == Mode::kLatency && num_threads_ > 1) {
    thread_pool_->submit(f, batch_size);
  } else {
    for (int i = 0; i < batch_size; i++) {
      f(i);
    }
  }
}

template <typename T>
void hnsw_lib<T>::save(const std::string& path_to_index) const
{
  appr_alg_->saveIndex(std::string(path_to_index));
}

template <typename T>
void hnsw_lib<T>::load(const std::string& path_to_index)
{
  if constexpr (std::is_same_v<T, float>) {
    if (metric_ == Metric::kInnerProduct) {
      space_ = std::make_shared<hnswlib::InnerProductSpace>(dim_);
    } else {
      space_ = std::make_shared<hnswlib::L2Space>(dim_);
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    space_ = std::make_shared<hnswlib::L2SpaceI<T>>(dim_);
  }

  appr_alg_ = std::make_shared<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    space_.get(), path_to_index);
}

template <typename T>
void hnsw_lib<T>::get_search_knn_results(const T* query,
                                         int k,
                                         algo_base::index_type* indices,
                                         float* distances) const
{
  auto result = appr_alg_->searchKnn(query, k);
  assert(result.size() >= static_cast<size_t>(k));

  for (int i = k - 1; i >= 0; --i) {
    indices[i]   = result.top().second;
    distances[i] = result.top().first;
    result.pop();
  }
}

};  // namespace cuvs::bench
