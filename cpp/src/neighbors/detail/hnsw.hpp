/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "../../core/nvtx.hpp"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <raft/core/logger.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <filesystem>
#include <memory>
#include <omp.h>
#include <random>
#include <thread>

namespace cuvs::neighbors::hnsw::detail {

// This is needed as hnswlib hardcodes the distance type to float
// or int32_t in certain places. However, we can solve uint8 or int8
// natively with the pacth cuVS applies. We could potentially remove
// all the hardcodes and propagate templates throughout hnswlib, but
// as of now it's not needed.
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
struct index_impl : index<T> {
 public:
  /**
   * @brief load a base-layer-only hnswlib index originally saved from a built CAGRA index
   *
   * @param[in] filepath path to the index
   * @param[in] dim dimensions of the training dataset
   * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
   * @param[in] hierarchy hierarchy used for upper HNSW layers
   */
  index_impl(int dim, cuvs::distance::DistanceType metric, HnswHierarchy hierarchy)
    : index<T>{dim, metric, hierarchy}
  {
    if constexpr (std::is_same_v<T, float>) {
      if (metric == cuvs::distance::DistanceType::L2Expanded) {
        space_ = std::make_unique<hnswlib::L2Space>(dim);
      } else if (metric == cuvs::distance::DistanceType::InnerProduct) {
        space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
      }
    } else if constexpr (std::is_same_v<T, std::int8_t> or std::is_same_v<T, std::uint8_t>) {
      if (metric == cuvs::distance::DistanceType::L2Expanded) {
        space_ = std::make_unique<hnswlib::L2SpaceI<T>>(dim);
      }
    }

    RAFT_EXPECTS(space_ != nullptr, "Unsupported metric type was used");
  }

  /**
  @brief Get hnswlib index
  */
  auto get_index() const -> void const* override { return appr_alg_.get(); }

  /**
  @brief Set ef for search
  */
  void set_ef(int ef) const override { appr_alg_->ef_ = ef; }

  /**
  @brief Set index
   */
  void set_index(std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>&& index)
  {
    appr_alg_ = std::move(index);
  }

  /**
  @brief Get space
   */
  auto get_space() const -> hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>*
  {
    return space_.get();
  }

 private:
  std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::unique_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;
};

template <typename T, HnswHierarchy hierarchy>
std::enable_if_t<hierarchy == HnswHierarchy::NONE, std::unique_ptr<index<T>>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::from_cagra<NONE>");
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0);
  auto uuid            = std::to_string(dist(rng));
  std::string filepath = "/tmp/" + uuid + ".bin";
  cuvs::neighbors::cagra::serialize_to_hnswlib(res, filepath, cagra_index, dataset);

  index<T>* hnsw_index = nullptr;
  int dim;
  if (dataset.has_value()) {
    dim = dataset.value().extent(1);
  } else {
    dim = cagra_index.dim();
  }

  cuvs::neighbors::hnsw::deserialize(res, params, filepath, dim, cagra_index.metric(), &hnsw_index);
  std::filesystem::remove(filepath);
  return std::unique_ptr<index<T>>(hnsw_index);
}

template <typename T, HnswHierarchy hierarchy>
std::enable_if_t<hierarchy == HnswHierarchy::CPU, std::unique_ptr<index<T>>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::from_cagra<CPU>");
  auto host_dataset = raft::make_host_matrix<T, int64_t>(0, 0);
  raft::host_matrix_view<const T, int64_t, raft::row_major> host_dataset_view(
    host_dataset.data_handle(), host_dataset.extent(0), host_dataset.extent(1));
  if (dataset.has_value()) {
    host_dataset_view = dataset.value();
  } else {
    // move dataset to host, remove padding
    auto cagra_dataset = cagra_index.dataset();
    RAFT_EXPECTS(cagra_dataset.size() > 0,
                 "Invalid CAGRA dataset of size 0, shape %zux%zu",
                 static_cast<size_t>(cagra_dataset.extent(0)),
                 static_cast<size_t>(cagra_dataset.extent(1)));
    host_dataset =
      raft::make_host_matrix<T, int64_t>(cagra_dataset.extent(0), cagra_dataset.extent(1));
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(host_dataset.data_handle(),
                                    sizeof(T) * host_dataset.extent(1),
                                    cagra_dataset.data_handle(),
                                    sizeof(T) * cagra_dataset.stride(0),
                                    sizeof(T) * host_dataset.extent(1),
                                    cagra_dataset.extent(0),
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(res)));
    raft::resource::sync_stream(res);
    host_dataset_view = host_dataset.view();
  }
  // build upper layers of hnsw index
  int dim         = host_dataset_view.extent(1);
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, cagra_index.metric(), hierarchy);
  auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(),
    host_dataset_view.extent(0),
    cagra_index.graph().extent(1) / 2,
    params.ef_construction);
  appr_algo->base_layer_init = false;  // tell hnswlib to build upper layers only
  auto num_threads           = params.num_threads == 0 ? omp_get_max_threads() : params.num_threads;
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < host_dataset_view.extent(0); i++) {
    appr_algo->addPoint((void*)(host_dataset_view.data_handle() + i * host_dataset_view.extent(1)),
                        i);
  }
  appr_algo->base_layer_init = true;  // reset to true to allow addition of new points

  // move cagra graph to host or access it from host if available
  auto host_graph_view = cagra_index.graph();
  auto host_graph      = raft::make_host_matrix<uint32_t, int64_t>(0, 0);
  if (!raft::is_host_accessible(raft::memory_type_from_pointer(host_graph_view.data_handle()))) {
    // copy cagra graph to host
    host_graph = raft::make_host_matrix<uint32_t, int64_t>(host_graph_view.extent(0),
                                                           host_graph_view.extent(1));
    raft::copy(host_graph.data_handle(),
               host_graph_view.data_handle(),
               host_graph_view.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    host_graph_view = host_graph.view();
  }

// copy cagra graph to hnswlib base layer
#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < static_cast<size_t>(host_graph_view.extent(0)); ++i) {
    auto hnsw_internal_id = appr_algo->label_lookup_.find(i)->second;
    auto ll_i             = appr_algo->get_linklist0(hnsw_internal_id);
    appr_algo->setListCount(ll_i, host_graph_view.extent(1));
    auto* data = (uint32_t*)(ll_i + 1);
    for (size_t j = 0; j < static_cast<size_t>(host_graph_view.extent(1)); ++j) {
      auto neighbor_internal_id = appr_algo->label_lookup_.find(host_graph(i, j))->second;
      data[j]                   = neighbor_internal_id;
    }
  }

  hnsw_index->set_index(std::move(appr_algo));
  return hnsw_index;
}

template <typename T, typename DistT>
int initialize_point_in_hnsw(hnswlib::HierarchicalNSW<DistT>* appr_algo,
                             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                             int64_t real_index,
                             int32_t curlevel)
{
  auto cur_c                        = appr_algo->cur_element_count++;
  appr_algo->element_levels_[cur_c] = curlevel;
  memset(appr_algo->data_level0_memory_ + cur_c * appr_algo->size_data_per_element_ +
           appr_algo->offsetLevel0_,
         0,
         appr_algo->size_data_per_element_);

  // Initialisation of the data and label
  memcpy(appr_algo->getExternalLabeLp(cur_c), &real_index, sizeof(hnswlib::labeltype));
  memcpy(appr_algo->getDataByInternalId(cur_c),
         dataset.data_handle() + real_index * dataset.extent(1),
         appr_algo->data_size_);

  if (curlevel) {
    appr_algo->linkLists_[cur_c] = (char*)malloc(appr_algo->size_links_per_element_ * curlevel + 1);
    if (appr_algo->linkLists_[cur_c] == nullptr)
      throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
    memset(appr_algo->linkLists_[cur_c], 0, appr_algo->size_links_per_element_ * curlevel + 1);
  }
  return cur_c;
}

template <typename T>
void all_neighbors_graph(raft::resources const& res,
                         raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                         raft::host_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
                         cuvs::distance::DistanceType metric)
{
  nn_descent::index_params nn_params;
  nn_params.graph_degree              = neighbors.extent(1);
  nn_params.intermediate_graph_degree = neighbors.extent(1) * 2;
  nn_params.metric                    = metric;
  nn_params.return_distances          = false;
  auto nn_index                       = nn_descent::build(res, nn_params, dataset, neighbors);
}

template <typename T, HnswHierarchy hierarchy>
std::enable_if_t<hierarchy == HnswHierarchy::GPU, std::unique_ptr<index<T>>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::from_cagra<GPU>");
  auto stream      = raft::resource::get_cuda_stream(res);
  auto num_threads = params.num_threads == 0 ? omp_get_max_threads() : params.num_threads;
  int64_t n_rows   = cagra_index.size();
  int64_t dim      = cagra_index.dim();

  // initialize HNSW index
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, cagra_index.metric(), hierarchy);
  auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(), n_rows, cagra_index.graph().extent(1) / 2, params.ef_construction);
  appr_algo->cur_element_count = n_rows;

  /* Note: NNSW data layout

  offsetLevel0_      = seems to always be 0, and it would break things otherwise
  size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
  offsetData_        = size_links_level0_;
  label_offset_      = size_links_level0_ + data_size_;

  get_linklist0(i)       = data_level0_memory_ + i * size_data_per_element_ + offsetLevel0_
  getDataByInternalId(i) = data_level0_memory_ + i * size_data_per_element_ + offsetData_
  getExternalLabeLp(i)   = data_level0_memory_ + i * size_data_per_element_ + label_offset_

  Hence the layout:
      2M x uint32_t  +   1 x uint32_t        dim x T    1 x size_t
     [linked list + linked list sizes]        [data]     [label]
  */

  bool device_copy;
  const T* source_dataset = nullptr;
  int64_t source_stride   = dim;
  if (dataset.has_value()) {
    device_copy    = false;
    source_dataset = dataset->data_handle();
  } else if (auto cagra_dataset = cagra_index.dataset(); cagra_dataset.data_handle() != nullptr) {
    device_copy    = true;
    source_dataset = cagra_dataset.data_handle();
    source_stride  = cagra_dataset.stride(0);
  } else {
    RAFT_FAIL("hnsw::from_cagra<GPU>: No dataset provided");
  }

  // Initialize linked lists
  auto& levels = appr_algo->element_levels_;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> block_scope(
      "parallel::initialize_data<%s>(%d threads)", device_copy ? "device" : "host", num_threads);
    /* Note: chunked strided cuda copy
    Ideally we want cudaMemcpy2DAsync to be completely non-blocking, running in parallel with the
    host loop. However, the cuda copy may sync with the host, because the host memory is paged (not
    pinned). The large the dataset, the larger the chance pages need to be swapped and thus would
    block. To mitigate this, run cuda copy in chunks. Yet, we cannot run a single copy per row,
    because parallel thread would fight for the cuda context and be forced to sleep.
    Hence the compromise: run one copy per bunch of operations and balance the work by making this
    stride not multiple of the num_threads.
    */
    const int64_t memcpy2s_stride = num_threads * 16 + 1;
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < n_rows; i++) {
      // clear the storage (TODO: it's not clear if this is necessary)
      memset(appr_algo->get_linklist0(i), 0, appr_algo->size_links_level0_);
      // copy the data section
      if (device_copy) {
        if ((i % memcpy2s_stride) == 0) {
          auto n_copy = std::min(memcpy2s_stride, n_rows - i);
          // Note: try to copy asynchronously and do the sync later
          RAFT_CUDA_TRY(cudaMemcpy2DAsync(appr_algo->getDataByInternalId(i),
                                          appr_algo->size_data_per_element_,
                                          source_dataset + i * source_stride,
                                          source_stride * sizeof(T),
                                          appr_algo->data_size_,
                                          n_copy,
                                          cudaMemcpyDefault,
                                          stream));
        }
      } else {
        memcpy(appr_algo->getDataByInternalId(i),
               source_dataset + i * source_stride,
               appr_algo->data_size_);
      }
      // As we build the index from scratch, we assign labels 1-1 in data order
      *appr_algo->getExternalLabeLp(i) = static_cast<hnswlib::labeltype>(i);
      int32_t curlevel                 = appr_algo->getRandomLevel(appr_algo->mult_);
      levels[i]                        = curlevel;
      if (curlevel) {
        appr_algo->linkLists_[i] = (char*)malloc(appr_algo->size_links_per_element_ * curlevel + 1);
        if (appr_algo->linkLists_[i] == nullptr)
          throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        memset(appr_algo->linkLists_[i], 0, appr_algo->size_links_per_element_ * curlevel + 1);
      }
    }
  }

  //   // Initialize HNSW dataset
  //   if (dataset.has_value()) {
  //     common::nvtx::range<common::nvtx::domain::cuvs> block_scope(
  //       "inititialize_dataset<host>(%d threads)", num_threads);
  //     auto host_dataset_view = dataset.value();
  // #pragma omp parallel for num_threads(num_threads)
  //     for (int64_t i = 0; i < n_rows; i++) {
  //       memcpy(appr_algo->getDataByInternalId(i),
  //              host_dataset_view.data_handle() + i * dim,
  //              appr_algo->data_size_);
  //     }
  //   } else if (auto cagra_dataset = cagra_index.dataset(); cagra_dataset.data_handle() !=
  //   nullptr) {
  //     common::nvtx::range<common::nvtx::domain::cuvs>
  //     block_scope("inititialize_dataset<device>");
  //     // Copy the data
  //     // NB: the stream sync is done later

  //   } else {
  //     RAFT_FAIL("hnsw::from_cagra<GPU>: No dataset provided");
  //   }

  // sort the points by levels
  // build histogram
  std::vector<size_t> hist;
  std::vector<size_t> order(n_rows);
  for (int64_t i = 0; i < n_rows; i++) {
    auto pt_level = levels[i];
    while (pt_level >= static_cast<int32_t>(hist.size()))
      hist.push_back(0);
    hist[pt_level]++;
  }

  // accumulate
  std::vector<size_t> offsets(hist.size() + 1, 0);
  for (size_t i = 0; i < hist.size() - 1; i++) {
    offsets[i + 1] = offsets[i] + hist[i];
  }

  // bucket sort
  for (int64_t i = 0; i < n_rows; i++) {
    auto pt_level              = levels[i];
    order[offsets[pt_level]++] = i;
  }

  // set last point of the highest level as the entry point
  appr_algo->enterpoint_node_ = order.back();
  appr_algo->maxlevel_        = hist.size() - 1;

  // Sync the stream only if we used it to copy the dataset;
  // we need the dataset in the following step
  if (device_copy) { raft::resource::sync_stream(res); }

  // iterate over the points in the descending order of their levels
  for (size_t pt_level = hist.size() - 1; pt_level >= 1; pt_level--) {
    common::nvtx::range<common::nvtx::domain::cuvs> level_scope("level %zu", pt_level);
    auto start_idx     = offsets[pt_level - 1];
    auto end_idx       = offsets[hist.size() - 1];
    auto num_pts       = end_idx - start_idx;
    auto neighbor_size = num_pts > appr_algo->M_ ? appr_algo->M_ : num_pts - 1;
    if (num_pts <= 1) {
      // this means only 1 point in the level
      continue;
    }

    // gather points from dataset to form query set on host
    auto host_query_set = raft::make_host_matrix<T, int64_t>(num_pts, dim);
    // TODO: Use `raft::matrix::gather` when available as a public API
    // Issue: https://github.com/rapidsai/raft/issues/2572
#pragma omp parallel for num_threads(num_threads)
    for (auto i = start_idx; i < end_idx; i++) {
      auto pt_id = order[i];
      std::copy(appr_algo->getDataByInternalId(pt_id),
                appr_algo->getDataByInternalId(pt_id) + dim,
                &host_query_set(i - start_idx, 0));
    }

    // find neighbors of the query set
    auto host_neighbors = raft::make_host_matrix<uint32_t, int64_t>(num_pts, neighbor_size);
    all_neighbors_graph(res,
                        raft::make_const_mdspan(host_query_set.view()),
                        host_neighbors.view(),
                        cagra_index.metric());

    {
      common::nvtx::range<common::nvtx::domain::cuvs> copy_scope(
        "get_linklist(%zu, %zu)", start_idx, end_idx);
      // add points to the HNSW index upper layers
#pragma omp parallel for num_threads(num_threads)
      for (auto i = start_idx; i < end_idx; i++) {
        auto pt_id  = order[i];
        auto ll_cur = appr_algo->get_linklist(pt_id, pt_level);
        appr_algo->setListCount(ll_cur, host_neighbors.extent(1));
        auto* data     = (uint32_t*)(ll_cur + 1);
        auto neighbors = &host_neighbors(i - start_idx, 0);
        for (auto j = 0; j < host_neighbors.extent(1); j++) {
          data[j] = order[neighbors[j] + start_idx];
        }
      }
    }
  }

  // move cagra graph to host or access it from host if available
  auto host_graph_view = cagra_index.graph();
  auto host_graph      = raft::make_host_matrix<uint32_t, int64_t>(0, 0);
  if (!raft::is_host_accessible(raft::memory_type_from_pointer(host_graph_view.data_handle()))) {
    // copy cagra graph to host
    host_graph = raft::make_host_matrix<uint32_t, int64_t>(host_graph_view.extent(0),
                                                           host_graph_view.extent(1));
    raft::copy(host_graph.data_handle(),
               host_graph_view.data_handle(),
               host_graph_view.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    host_graph_view = host_graph.view();
  }

  {
    common::nvtx::range<common::nvtx::domain::cuvs> copy_scope("get_linklist0");
// copy cagra graph to hnswlib base layer
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < static_cast<size_t>(host_graph_view.extent(0)); ++i) {
      auto ll_i = appr_algo->get_linklist0(i);
      appr_algo->setListCount(ll_i, host_graph_view.extent(1));
      auto* data = (uint32_t*)(ll_i + 1);
      for (size_t j = 0; j < static_cast<size_t>(host_graph_view.extent(1)); ++j) {
        data[j] = host_graph_view(i, j);
      }
    }
  }

  hnsw_index->set_index(std::move(appr_algo));
  return hnsw_index;
}

template <typename T>
std::unique_ptr<index<T>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  if (params.hierarchy == HnswHierarchy::NONE) {
    return from_cagra<T, HnswHierarchy::NONE>(res, params, cagra_index, dataset);
  } else if (params.hierarchy == HnswHierarchy::CPU) {
    return from_cagra<T, HnswHierarchy::CPU>(res, params, cagra_index, dataset);
  } else if (params.hierarchy == HnswHierarchy::GPU) {
    return from_cagra<T, HnswHierarchy::GPU>(res, params, cagra_index, dataset);
  } else {
    RAFT_FAIL("Unsupported hierarchy type");
  }
}

template <typename T>
void extend(raft::resources const& res,
            const extend_params& params,
            raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,
            index<T>& idx)
{
  auto* hnswlib_index = reinterpret_cast<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>*>(
    const_cast<void*>(idx.get_index()));
  auto current_element_count = hnswlib_index->getCurrentElementCount();
  auto new_element_count     = additional_dataset.extent(0);
  auto num_threads           = params.num_threads == 0 ? omp_get_max_threads() : params.num_threads;

  hnswlib_index->resizeIndex(current_element_count + new_element_count);
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < additional_dataset.extent(0); i++) {
    hnswlib_index->addPoint(
      (void*)(additional_dataset.data_handle() + i * additional_dataset.extent(1)),
      current_element_count + i);
  }
}

template <typename T>
void get_search_knn_results(hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type> const* idx,
                            const T* query,
                            int k,
                            uint64_t* indices,
                            float* distances)
{
  auto result = idx->searchKnn(query, k);
  assert(result.size() >= static_cast<size_t>(k));

  for (int i = k - 1; i >= 0; --i) {
    indices[i]   = result.top().second;
    distances[i] = result.top().first;
    result.pop();
  }
}

template <typename T>
void search(raft::resources const& res,
            const search_params& params,
            const index<T>& idx,
            raft::host_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances)
{
  RAFT_EXPECTS(queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
               "Number of rows in output neighbors and distances matrices must equal the number of "
               "queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");
  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  idx.set_ef(params.ef);
  auto const* hnswlib_index =
    reinterpret_cast<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type> const*>(
      idx.get_index());

  // when num_threads == 0, automatically maximize parallelism
  if (params.num_threads) {
#pragma omp parallel for num_threads(params.num_threads)
    for (int64_t i = 0; i < queries.extent(0); ++i) {
      get_search_knn_results(hnswlib_index,
                             queries.data_handle() + i * queries.extent(1),
                             neighbors.extent(1),
                             neighbors.data_handle() + i * neighbors.extent(1),
                             distances.data_handle() + i * distances.extent(1));
    }
  } else {
#pragma omp parallel for
    for (int64_t i = 0; i < queries.extent(0); ++i) {
      get_search_knn_results(hnswlib_index,
                             queries.data_handle() + i * queries.extent(1),
                             neighbors.extent(1),
                             neighbors.data_handle() + i * neighbors.extent(1),
                             distances.data_handle() + i * distances.extent(1));
    }
  }
}

template <typename T>
void serialize(raft::resources const& res, const std::string& filename, const index<T>& idx)
{
  auto* hnswlib_index = reinterpret_cast<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>*>(
    const_cast<void*>(idx.get_index()));
  hnswlib_index->saveIndex(filename);
}

template <typename T>
void deserialize(raft::resources const& res,
                 const index_params& params,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<T>** idx)
{
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, metric, params.hierarchy);
  auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(), filename);
  if (params.hierarchy == HnswHierarchy::NONE) { appr_algo->base_layer_only = true; }
  hnsw_index->set_index(std::move(appr_algo));
  *idx = hnsw_index.release();
}

}  // namespace cuvs::neighbors::hnsw::detail
