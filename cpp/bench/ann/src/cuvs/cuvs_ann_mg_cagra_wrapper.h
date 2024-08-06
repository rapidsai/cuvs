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
#include "cuvs_cagra_wrapper.h"
#include <cuvs/neighbors/ann_mg.hpp>

namespace cuvs::bench {

enum class AllocatorType;
enum class CagraBuildAlgo;

template <typename T, typename IdxT>
class cuvs_ann_mg_cagra : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;

  struct build_param {
    cuvs::neighbors::cagra::mg_index_params cagra_params;
    CagraBuildAlgo algo;
    std::optional<cuvs::neighbors::nn_descent::index_params> nn_descent_params = std::nullopt;
    std::optional<float> ivf_pq_refine_rate                                    = std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::index_params> ivf_pq_build_params   = std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::search_params> ivf_pq_search_params = std::nullopt;
  };

  struct search_param : public cuvs::bench::cuvs_cagra<T, IdxT>::search_param {
    cuvs::neighbors::mg::sharded_merge_mode merge_mode;
  };

  cuvs_ann_mg_cagra(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim), index_params_(param)
  {
    index_params_.cagra_params.metric         = parse_metric_type(metric);
    index_params_.ivf_pq_build_params->metric = parse_metric_type(metric);
    clique_ = std::make_shared<cuvs::neighbors::mg::nccl_clique>();
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param) override;

  void set_search_dataset(const T* dataset, size_t nrow) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    const auto& handle = clique_->set_current_device_to_root_rank();
    auto stream        = raft::resource::get_cuda_stream(handle);
    return stream;
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  void save_to_hnswlib(const std::string& file) const;
  std::unique_ptr<algo<T>> copy() override;

 private:
  std::shared_ptr<cuvs::neighbors::mg::nccl_clique> clique_;
  float refine_ratio_;
  build_param index_params_;
  cuvs::neighbors::cagra::search_params search_params_;
  cuvs::neighbors::mg::sharded_merge_mode merge_mode_;
  std::shared_ptr<
    cuvs::neighbors::mg::ann_mg_index<cuvs::neighbors::cagra::index<T, IdxT>, T, IdxT>>
    index_;
};

template <typename T, typename IdxT>
void cuvs_ann_mg_cagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_extents = raft::make_extents<IdxT>(nrow, dim_);

  auto& params = index_params_.cagra_params;
  if (index_params_.algo == CagraBuildAlgo::kIvfPq) {
    auto pq_params =
      cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(dataset_extents, params.metric);
    if (index_params_.ivf_pq_build_params) {
      pq_params.build_params = *index_params_.ivf_pq_build_params;
    }
    if (index_params_.ivf_pq_search_params) {
      pq_params.search_params = *index_params_.ivf_pq_search_params;
    }
    if (index_params_.ivf_pq_refine_rate) {
      pq_params.refinement_rate = *index_params_.ivf_pq_refine_rate;
    }
    params.graph_build_params = pq_params;
  } else if (index_params_.algo == CagraBuildAlgo::kNnDescent) {
    auto nn_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      params.intermediate_graph_degree);
    if (index_params_.nn_descent_params) { nn_params = *index_params_.nn_descent_params; }
    params.graph_build_params = nn_params;
  }

  auto dataset_view =
    raft::make_host_matrix_view<const T, int64_t, raft::row_major>(dataset, nrow, dim_);
  const auto& handle = clique_->set_current_device_to_root_rank();
  auto idx           = cuvs::neighbors::mg::build(handle, *clique_, params, dataset_view);
  index_             = std::make_shared<
    cuvs::neighbors::mg::ann_mg_index<cuvs::neighbors::cagra::index<T, IdxT>, T, IdxT>>(
    std::move(idx));
}

inline auto allocator_to_string(AllocatorType mem_type) -> std::string;

template <typename T, typename IdxT>
void cuvs_ann_mg_cagra<T, IdxT>::set_search_param(const search_param_base& param)
{
  auto sp        = dynamic_cast<const search_param&>(param);
  search_params_ = sp.p;
  merge_mode_    = sp.merge_mode;
  refine_ratio_  = sp.refine_ratio;

  /*
  if (sp.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = sp.graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr        = get_mr(graph_mem_);
    auto new_graph = raft::make_device_mdarray<IdxT, int64_t>(
      handle_, mr, raft::make_extents<int64_t>(index_->graph().extent(0), index_->graph_degree()));

    raft::copy(new_graph.data_handle(),
               index_->graph().data_handle(),
               index_->graph().size(),
               raft::resource::get_cuda_stream(handle_));

    index_->update_graph(handle_, make_const_mdspan(new_graph.view()));
    // update_graph() only stores a view in the index. We need to keep the graph object alive.
    *graph_ = std::move(new_graph);
  }

  if (sp.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = sp.dataset_mem;

    // First free up existing memory
    *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                   allocator_to_string(dataset_mem_).c_str());

    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::mg::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);

    need_dataset_update_ = false;
  }
  */
}

template <typename T, typename IdxT>
void cuvs_ann_mg_cagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  /*
  using ds_idx_type = decltype(index_->data().n_rows());
  bool is_vpq =
    dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
    dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
  // It can happen that we are re-using a previous algo object which already has
  // the dataset set. Check if we need update.
  if (static_cast<size_t>(input_dataset_v_->extent(0)) != nrow ||
      input_dataset_v_->data_handle() != dataset) {
    *input_dataset_v_ = raft::make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
    need_dataset_update_ = !is_vpq;  // ignore update if this is a VPQ dataset.
  }
  */
}

template <typename T, typename IdxT>
void cuvs_ann_mg_cagra<T, IdxT>::save(const std::string& file) const
{
  const auto& handle = clique_->set_current_device_to_root_rank();
  cuvs::neighbors::mg::serialize(handle, *clique_, *index_, file);
}

template <typename T, typename IdxT>
void cuvs_ann_mg_cagra<T, IdxT>::load(const std::string& file)
{
  const auto& handle = clique_->set_current_device_to_root_rank();
  index_             = std::make_shared<
    cuvs::neighbors::mg::ann_mg_index<cuvs::neighbors::cagra::index<T, IdxT>, T, IdxT>>(
    std::move(cuvs::neighbors::mg::deserialize_cagra<T, IdxT>(handle, *clique_, file)));
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_ann_mg_cagra<T, IdxT>::copy()
{
  return std::make_unique<cuvs_ann_mg_cagra<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_ann_mg_cagra<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  auto queries_view =
    raft::make_host_matrix_view<const T, int64_t, raft::row_major>(queries, batch_size, dim_);
  auto neighbors_view =
    raft::make_host_matrix_view<IdxT, int64_t, raft::row_major>((IdxT*)neighbors, batch_size, k);
  auto distances_view =
    raft::make_host_matrix_view<float, int64_t, raft::row_major>(distances, batch_size, k);

  const auto& handle = clique_->set_current_device_to_root_rank();
  cuvs::neighbors::mg::search(handle,
                              *clique_,
                              *index_,
                              search_params_,
                              queries_view,
                              neighbors_view,
                              distances_view,
                              merge_mode_);
}

template <typename T, typename IdxT>
void cuvs_ann_mg_cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    throw std::runtime_error("refinement not supported");
  }
}
}  // namespace cuvs::bench
