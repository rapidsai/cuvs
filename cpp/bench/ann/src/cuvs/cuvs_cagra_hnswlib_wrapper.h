/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <raft/core/logger.hpp>

#include <chrono>
#include <memory>

namespace cuvs::bench {

class s_res {
 public:
  using large_mr_type = rmm::mr::managed_memory_resource;
  using mr_type       = rmm::mr::pool_memory_resource<large_mr_type>;

  s_res()
  try
    : orig_resource_{rmm::mr::get_current_device_resource()},
      large_mr_{},
      resource_(&large_mr_, 1024 * 1024 * 1024ull) {
    rmm::mr::set_current_device_resource(&large_mr_);
  } catch (const std::exception& e) {
    auto cuda_status = cudaGetLastError();
    size_t free      = 0;
    size_t total     = 0;
    RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
    RAFT_FAIL(
      "Failed to initialize shared raft resources (NB: latest cuda status = %s, free memory = %zu, "
      "total memory = %zu): %s",
      cudaGetErrorName(cuda_status),
      free,
      total,
      e.what());
  }

  s_res(s_res&&)                               = delete;
  auto operator=(s_res&&) -> s_res&            = delete;
  s_res(const s_res& res)                      = delete;
  auto operator=(const s_res& other) -> s_res& = delete;

  ~s_res() noexcept { rmm::mr::set_current_device_resource(orig_resource_); }

  auto get_small_memory_resource() noexcept
  {
    return static_cast<rmm::mr::device_memory_resource*>(&resource_);
  }

  auto get_large_memory_resource() noexcept
  {
    return static_cast<rmm::mr::device_memory_resource*>(&large_mr_);
  }

 private:
  rmm::mr::device_memory_resource* orig_resource_;
  large_mr_type large_mr_;
  mr_type resource_;
};

class cagra_build_raft_resources {
 public:
  /**
   * This constructor has the shared state passed unmodified but creates the local state anew.
   * It's used by the copy constructor.
   */
  explicit cagra_build_raft_resources(const std::shared_ptr<s_res>& shared_res)
    : shared_res_{shared_res},
      res_{std::make_unique<raft::device_resources>(
        rmm::cuda_stream_view(get_stream_from_global_pool()))}
  {
    // set the large workspace resource to the raft handle, but without the deleter
    // (this resource is managed by the shared_res).
    raft::resource::set_workspace_resource(
      *res_,
      std::shared_ptr<rmm::mr::device_memory_resource>(shared_res_->get_small_memory_resource(),
                                                       raft::void_op{}));
    raft::resource::set_large_workspace_resource(
      *res_,
      std::shared_ptr<rmm::mr::device_memory_resource>(shared_res_->get_large_memory_resource(),
                                                       raft::void_op{}));
  }

  /** Default constructor creates all resources anew. */
  cagra_build_raft_resources() : cagra_build_raft_resources{std::make_shared<s_res>()} {}

  cagra_build_raft_resources(cagra_build_raft_resources&&);
  auto operator=(cagra_build_raft_resources&&) -> cagra_build_raft_resources&;
  ~cagra_build_raft_resources() = default;
  cagra_build_raft_resources(const cagra_build_raft_resources& res)
    : cagra_build_raft_resources{res.shared_res_}
  {
  }
  auto operator=(const cagra_build_raft_resources& other) -> cagra_build_raft_resources&
  {
    this->shared_res_ = other.shared_res_;
    return *this;
  }

  operator raft::resources&() noexcept { return *res_; }              // NOLINT
  operator const raft::resources&() const noexcept { return *res_; }  // NOLINT

  /** Get the main stream */
  [[nodiscard]] auto get_sync_stream() const noexcept
  {
    return raft::resource::get_cuda_stream(*res_);
  }

 private:
  /** The resources shared among multiple raft handles / threads. */
  std::shared_ptr<s_res> shared_res_;
  /**
   * Until we make the use of copies of raft::resources thread-safe, each benchmark wrapper must
   * have its own copy of it.
   */
  std::unique_ptr<raft::device_resources> res_ = std::make_unique<raft::device_resources>();
};

template <typename T, typename IdxT>
class cuvs_cagra_hnswlib : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct build_param {
    using cagra_wrapper_params = typename cuvs_cagra<T, IdxT>::build_param;
    cagra_wrapper_params cagra_build_params;
    cuvs::neighbors::hnsw::index_params hnsw_index_params;
  };

  struct search_param : public search_param_base {
    cuvs::neighbors::hnsw::search_params hnsw_search_param;
  };

  cuvs_cagra_hnswlib(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim), build_param_{param}
  {
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  [[nodiscard]] auto uses_stream() const noexcept -> bool override
  {
    // there's no need to synchronize with the GPU neither on build nor on search
    return false;
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
  cagra_build_raft_resources handle_{};
  build_param build_param_;
  search_param search_param_;
  std::shared_ptr<cuvs::neighbors::hnsw::index<T>> hnsw_index_;
};

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::build(const T* dataset, size_t nrow)
{
  // when the data set is on host, we can pass it directly to HNSW
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;

  // re-use the CAGRA wrapper to parse build params
  auto bps = build_param_.cagra_build_params;
  // Not very conveniently, the CAGRA wrapper resolves parameters after the dataset shape is known,
  // so it takes a lambda to do it. Even though we know the shape, we want to use the wrapper as-is,
  // so we just modify that lambda.
  bps.cagra_params = [dataset_is_on_host, orig_cagra_params = bps.cagra_params](
                       auto dataset_extents, auto metric) {
    auto params                    = orig_cagra_params(dataset_extents, metric);
    params.attach_dataset_on_build = !dataset_is_on_host;
    return params;
  };
  cuvs_cagra<T, IdxT> cagra_wrapper{this->metric_, this->dim_, bps};

  // build the CAGRA index
  cagra_wrapper.build(dataset, nrow);
  auto& cagra_index = *cagra_wrapper.get_index();

  // pass the dataset directly to HNSW if it's on the host
  std::optional<raft::host_matrix_view<const T, int64_t>> opt_dataset_view = std::nullopt;
  if (dataset_is_on_host) {
    opt_dataset_view.emplace(
      raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, this->dim_));
  }

  // convert the index to HNSW format
  hnsw_index_ = cuvs::neighbors::hnsw::from_cagra(
    handle_, build_param_.hnsw_index_params, cagra_index, opt_dataset_view);
}

template <typename T, typename IdxT>
void cuvs_cagra_hnswlib<T, IdxT>::set_search_param(const search_param_base& param_,
                                                   const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
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
