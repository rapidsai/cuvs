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
#include "../common/util.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/index_io.h>
#include <omp.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

auto parse_metric_type(cuvs::bench::Metric metric) -> faiss::MetricType
{
  if (metric == cuvs::bench::Metric::kInnerProduct) {
    return faiss::METRIC_INNER_PRODUCT;
  } else if (metric == cuvs::bench::Metric::kEuclidean) {
    return faiss::METRIC_L2;
  } else {
    throw std::runtime_error("faiss supports only metric type of inner product and L2");
  }
}

// note BLAS library can still use multi-threading, and
// setting environment variable like OPENBLAS_NUM_THREADS can control it
class omp_single_thread_scope {
 public:
  omp_single_thread_scope()
  {
    max_threads_ = omp_get_max_threads();
    omp_set_num_threads(1);
  }
  ~omp_single_thread_scope()
  {
    // the best we can do
    omp_set_num_threads(max_threads_);
  }

 private:
  int max_threads_;
};

}  // namespace

namespace cuvs::bench {

template <typename T>
class faiss_gpu : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    int nprobe;
    float refine_ratio = 1.0;
    [[nodiscard]] auto needs_dataset() const -> bool override { return refine_ratio > 1.0f; }
  };

  struct build_param {
    int nlist = 1;
    int ratio = 2;
  };

  faiss_gpu(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim),
      gpu_resource_{std::make_shared<faiss::gpu::StandardGpuResources>()},
      metric_type_(parse_metric_type(metric)),
      nlist_{param.nlist},
      training_sample_fraction_{1.0 / double(param.ratio)}
  {
    static_assert(std::is_same_v<T, float>, "faiss support only float type");
    cudaGetDevice(&device_);
  }

  void build(const T* dataset, size_t nrow) final;

  virtual void set_search_param(const search_param_base& param) {}

  void set_search_dataset(const T* dataset, size_t nrow) override { dataset_ = dataset; }

  // TODO(snanditale): if the number of results is less than k, the remaining elements of
  // 'neighbors' will be filled with (size_t)-1
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const final;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return gpu_resource_->getDefaultStream(device_);
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    // to enable building big dataset which is larger than GPU memory
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 protected:
  template <typename GpuIndex, typename CpuIndex>
  void save_(const std::string& file) const;  // NOLINT

  template <typename GpuIndex, typename CpuIndex>
  void load_(const std::string& file);  // NOLINT

  /** [NOTE Multithreading]
   *
   * `gpu_resource_` is a shared resource:
   *   1. It uses a shared_ptr under the hood, so the copies of it refer to the same
   *      resource implementation instance
   *   2. GpuIndex is probably keeping a reference to it, as it's passed to the constructor
   *
   * To avoid copying the index (database) in each thread, we make both the index and
   * the gpu_resource shared.
   * This means faiss GPU streams are possibly shared among the CPU threads;
   * the throughput search mode may be inaccurate.
   *
   * WARNING: we haven't investigated whether faiss::gpu::GpuIndex or
   * faiss::gpu::StandardGpuResources are thread-safe.
   *
   */
  mutable std::shared_ptr<faiss::gpu::StandardGpuResources> gpu_resource_;
  std::shared_ptr<faiss::gpu::GpuIndex> index_;
  std::shared_ptr<faiss::IndexRefineFlat> index_refine_{nullptr};
  faiss::MetricType metric_type_;
  int nlist_;
  int device_;
  double training_sample_fraction_;
  std::shared_ptr<faiss::SearchParameters> search_params_;
  const T* dataset_;
  float refine_ratio_ = 1.0;
};

template <typename T>
void faiss_gpu<T>::build(const T* dataset, size_t nrow)
{
  omp_single_thread_scope omp_single_thread;
  auto index_ivf = dynamic_cast<faiss::gpu::GpuIndexIVF*>(index_.get());
  if (index_ivf != nullptr) {
    // set the min/max training size for clustering to use the whole provided training set.
    double trainset_size       = training_sample_fraction_ * static_cast<double>(nrow);
    double points_per_centroid = trainset_size / static_cast<double>(nlist_);
    int max_ppc                = std::ceil(points_per_centroid);
    int min_ppc                = std::floor(points_per_centroid);
    if (min_ppc < index_ivf->cp.min_points_per_centroid) {
      log_warn(
        "The suggested training set size %zu (data size %zu, training sample ratio %f) yields %d "
        "points per cluster (n_lists = %d). This is smaller than the FAISS default "
        "min_points_per_centroid = %d.",
        static_cast<size_t>(trainset_size),
        nrow,
        training_sample_fraction_,
        min_ppc,
        nlist_,
        index_ivf->cp.min_points_per_centroid);
    }
    index_ivf->cp.max_points_per_centroid = max_ppc;
    index_ivf->cp.min_points_per_centroid = min_ppc;
  }
  index_->train(nrow, dataset);  // faiss::gpu::GpuIndexFlat::train() will do nothing
  assert(index_->is_trained);
  index_->add(nrow, dataset);
}

template <typename T>
void faiss_gpu<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(sizeof(size_t) == sizeof(faiss::idx_t),
                "sizes of size_t and faiss::idx_t are different");

  if (this->refine_ratio_ > 1.0) {
    // TODO(snanditale): FAISS changed their search APIs to accept the search parameters as a struct
    // object but their refine API doesn't allow the struct to be passed in. Once this is fixed, we
    // need to re-enable refinement below
    // index_refine_->search(batch_size, queries, k, distances,
    // reinterpret_cast<faiss::idx_t*>(neighbors), this->search_params_.get()); Related FAISS issue:
    // https://github.com/facebookresearch/faiss/issues/3118
    throw std::runtime_error(
      "FAISS doesn't support refinement in their new APIs so this feature is disabled in the "
      "benchmarks for the time being.");
  } else {
    index_->search(batch_size,
                   queries,
                   k,
                   distances,
                   reinterpret_cast<faiss::idx_t*>(neighbors),
                   this->search_params_.get());
  }
}

template <typename T>
template <typename GpuIndex, typename CpuIndex>
void faiss_gpu<T>::save_(const std::string& file) const
{
  omp_single_thread_scope omp_single_thread;

  auto cpu_index = std::make_unique<CpuIndex>();
  dynamic_cast<GpuIndex*>(index_.get())->copyTo(cpu_index.get());
  faiss::write_index(cpu_index.get(), file.c_str());
}

template <typename T>
template <typename GpuIndex, typename CpuIndex>
void faiss_gpu<T>::load_(const std::string& file)
{
  omp_single_thread_scope omp_single_thread;

  std::unique_ptr<CpuIndex> cpu_index(dynamic_cast<CpuIndex*>(faiss::read_index(file.c_str())));
  assert(cpu_index);

  try {
    dynamic_cast<GpuIndex*>(index_.get())->copyFrom(cpu_index.get());

  } catch (const std::exception& e) {
    std::cout << "Error loading index file: " << std::string(e.what()) << std::endl;
  }
}

template <typename T>
class faiss_gpu_ivf_flat : public faiss_gpu<T> {
 public:
  using typename faiss_gpu<T>::build_param;
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_ivf_flat(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = this->device_;
    this->index_  = std::make_shared<faiss::gpu::GpuIndexIVFFlat>(
      this->gpu_resource_.get(), dim, param.nlist, this->metric_type_, config);
  }

  void set_search_param(const search_param_base& param) override
  {
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;
    this->search_params_       = std::make_shared<faiss::IVFSearchParameters>(faiss_search_params);
    this->refine_ratio_        = sp.refine_ratio;
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFFlat, faiss::IndexIVFFlat>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFFlat, faiss::IndexIVFFlat>(file);
  }
  std::unique_ptr<algo<T>> copy() override
  {
    return std::make_unique<faiss_gpu_ivf_flat<T>>(*this);
  };
};

template <typename T>
class faiss_gpu_ivfpq : public faiss_gpu<T> {
 public:
  struct build_param : public faiss_gpu<T>::build_param {
    int m;
    bool use_float16;
    bool use_precomputed;
  };
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_ivfpq(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.useFloat16LookupTables = param.use_float16;
    config.usePrecomputedTables   = param.use_precomputed;
    config.device                 = this->device_;

    this->index_ =
      std::make_shared<faiss::gpu::GpuIndexIVFPQ>(this->gpu_resource_.get(),
                                                  dim,
                                                  param.nlist,
                                                  param.m,
                                                  8,  // FAISS only supports bitsPerCode=8
                                                  this->metric_type_,
                                                  config);
  }

  void set_search_param(const search_param_base& param) override
  {
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);
    this->refine_ratio_ = sp.refine_ratio;
    faiss::IVFPQSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;

    this->search_params_ = std::make_shared<faiss::IVFPQSearchParameters>(faiss_search_params);

    if (sp.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_shared<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = sp.refine_ratio;
    }
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFPQ, faiss::IndexIVFPQ>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFPQ, faiss::IndexIVFPQ>(file);
  }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<faiss_gpu_ivfpq<T>>(*this); };
};

// TODO(snanditale): Enable this in cmake
//  ref: https://github.com/rapidsai/raft/issues/1876
template <typename T>
class faiss_gpu_ivfsq : public faiss_gpu<T> {
 public:
  struct build_param : public faiss_gpu<T>::build_param {
    std::string quantizer_type;
  };
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_ivfsq(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    faiss::ScalarQuantizer::QuantizerType qtype;
    if (param.quantizer_type == "fp16") {
      qtype = faiss::ScalarQuantizer::QT_fp16;
    } else if (param.quantizer_type == "int8") {
      qtype = faiss::ScalarQuantizer::QT_8bit;
    } else {
      throw std::runtime_error("faiss_gpu_ivfsq supports only fp16 and int8 but got " +
                               param.quantizer_type);
    }

    faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
    config.device = this->device_;
    this->index_  = std::make_shared<faiss::gpu::GpuIndexIVFScalarQuantizer>(
      this->gpu_resource_.get(), dim, param.nlist, qtype, this->metric_type_, true, config);
  }

  void set_search_param(const search_param_base& param) override
  {
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;

    this->search_params_ = std::make_shared<faiss::IVFSearchParameters>(faiss_search_params);
    this->refine_ratio_  = sp.refine_ratio;
    if (sp.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_shared<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = sp.refine_ratio;
    }
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>(
      file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>(
      file);
  }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<faiss_gpu_ivfsq<T>>(*this); };
};

template <typename T>
class faiss_gpu_flat : public faiss_gpu<T> {
 public:
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_flat(Metric metric, int dim)
    : faiss_gpu<T>(metric, dim, typename faiss_gpu<T>::build_param{})
  {
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = this->device_;
    this->index_  = std::make_shared<faiss::gpu::GpuIndexFlat>(
      this->gpu_resource_.get(), dim, this->metric_type_, config);
  }
  void set_search_param(const search_param_base& param) override
  {
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);

    this->search_params_ = std::make_shared<faiss::SearchParameters>();
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<faiss_gpu_flat<T>>(*this); };
};

}  // namespace cuvs::bench
