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

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <filesystem>
#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>
#include <memory>
#include <random>
#include <thread>

namespace cuvs::neighbors::hnsw::detail {

// Multithreaded executor
// The helper function is copied from the hnswlib repository
// as for some reason, adding vectors to the hnswlib index does not
// work well with omp parallel for
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
  if (numThreads <= 0) { numThreads = std::thread::hardware_concurrency(); }

  if (numThreads == 1) {
    for (size_t id = start; id < end; id++) {
      fn(id, 0);
    }
  } else {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
      threads.push_back(std::thread([&, threadId] {
        while (true) {
          size_t id = current.fetch_add(1);

          if (id >= end) { break; }

          try {
            fn(id, threadId);
          } catch (...) {
            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
            lastException = std::current_exception();
            /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
            current = end;
            break;
          }
        }
      }));
    }
    for (auto& thread : threads) {
      thread.join();
    }
    if (lastException) { std::rethrow_exception(lastException); }
  }
}

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
  const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index)
{
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0);
  auto uuid            = std::to_string(dist(rng));
  std::string filepath = "/tmp/" + uuid + ".bin";
  cuvs::neighbors::cagra::serialize_to_hnswlib(res, filepath, cagra_index);

  index<T>* hnsw_index = nullptr;
  cuvs::neighbors::hnsw::deserialize(
    res, params, filepath, cagra_index.dim(), cagra_index.metric(), &hnsw_index);
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
  auto host_dataset = raft::make_host_matrix<T, int64_t>(0, 0);
  raft::host_matrix_view<const T, int64_t, raft::row_major> host_dataset_view(
    host_dataset.data_handle(), host_dataset.extent(0), host_dataset.extent(1));
  if (dataset.has_value()) {
    host_dataset_view = dataset.value();
  } else {
    // move dataset to host, remove padding
    auto cagra_dataset = cagra_index.dataset();
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
  auto hnsw_index =
    std::make_unique<index_impl<T>>(cagra_index.dim(), cagra_index.metric(), hierarchy);
  auto appr_algo = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(),
    host_dataset_view.extent(0),
    cagra_index.graph().extent(1) / 2,
    params.ef_construction);
  appr_algo->base_layer_init = false;  // tell hnswlib to build upper layers only
  ParallelFor(0, host_dataset_view.extent(0), params.num_threads, [&](size_t i, size_t threadId) {
    appr_algo->addPoint((void*)(host_dataset_view.data_handle() + i * host_dataset_view.extent(1)),
                        i);
  });
  appr_algo->base_layer_init = true;  // reset to true to allow addition of new points

  // move cagra graph to host
  auto graph = cagra_index.graph();
  auto host_graph =
    raft::make_host_matrix<uint32_t, int64_t, raft::row_major>(graph.extent(0), graph.extent(1));
  raft::copy(host_graph.data_handle(),
             graph.data_handle(),
             graph.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

// copy cagra graph to hnswlib base layer
#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(host_graph.extent(0)); ++i) {
    auto ll_i = appr_algo->get_linklist0(i);
    appr_algo->setListCount(ll_i, host_graph.extent(1));
    auto* data = (uint32_t*)(ll_i + 1);
    for (size_t j = 0; j < static_cast<size_t>(host_graph.extent(1)); ++j) {
      data[j] = host_graph(i, j);
    }
  }

  hnsw_index->set_index(std::move(appr_algo));
  return hnsw_index;
}

int initialize_point_in_hnsw(hnswlib::HierarchicalNSW<float>* appr_algo,
                             raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
                             int64_t real_index,
                             int curlevel)
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

template <typename T, HnswHierarchy hierarchy>
std::enable_if_t<hierarchy == HnswHierarchy::GPU, std::unique_ptr<index<T>>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  auto host_dataset = raft::make_host_matrix<T, int64_t>(0, 0);
  raft::host_matrix_view<const T, int64_t, raft::row_major> host_dataset_view(
    host_dataset.data_handle(), host_dataset.extent(0), host_dataset.extent(1));
  if (dataset.has_value()) {
    host_dataset_view = dataset.value();
  } else {
    // move dataset to host, remove padding
    auto cagra_dataset = cagra_index.dataset();
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

  // initialize hnsw index
  auto hnsw_index =
    std::make_unique<index_impl<T>>(cagra_index.dim(), cagra_index.metric(), hierarchy);
  auto appr_algo = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(),
    host_dataset_view.extent(0),
    cagra_index.graph().extent(1) / 2,
    params.ef_construction);

  // assign a level to each point and initialize the points in hnsw
  std::vector<size_t> levels(host_dataset_view.extent(0));
  std::vector<uint32_t> hnsw_internal_ids(host_dataset_view.extent(0));

#pragma omp parallel for
  for (int64_t i = 0; i < host_dataset_view.extent(0); i++) {
    levels[i] = appr_algo->getRandomLevel(appr_algo->mult_) + 1;
    hnsw_internal_ids[i] =
      initialize_point_in_hnsw(appr_algo.get(), host_dataset_view, i, levels[i] - 1);
  }

  // sort the points by levels
  // build histogram
  std::vector<size_t> hist;
  std::vector<size_t> order(host_dataset_view.extent(0));
  for (int64_t i = 0; i < host_dataset_view.extent(0); i++) {
    auto pt_level = levels[i] - 1;
    while (pt_level >= hist.size())
      hist.push_back(0);
    hist[pt_level]++;
  }

  // accumulate
  std::vector<size_t> offsets(hist.size() + 1, 0);
  for (size_t i = 0; i < hist.size() - 1; i++) {
    offsets[i + 1] = offsets[i] + hist[i];
  }

  // bucket sort
  for (int64_t i = 0; i < host_dataset_view.extent(0); i++) {
    auto pt_level              = levels[i] - 1;
    order[offsets[pt_level]++] = i;
  }

  // set last point of the highest level as the entry point
  appr_algo->enterpoint_node_ = hnsw_internal_ids[order.back()];
  appr_algo->maxlevel_        = hist.size() - 1;

  // iterate over the points in the descending order of their levels
  for (size_t pt_level = hist.size() - 1; pt_level >= 1; pt_level--) {
    auto start_idx     = offsets[pt_level - 1];
    auto end_idx       = offsets[hist.size() - 1];
    auto num_pts       = end_idx - start_idx;
    auto neighbor_size = num_pts > appr_algo->M_ ? appr_algo->M_ : num_pts;
    if (neighbor_size < 2) {
      // this means only 1 point in the level
      continue;
    }

    // gather points from dataset to form query set on device
    auto query_set =
      raft::make_device_matrix<T, int64_t>(res, num_pts, host_dataset_view.extent(1));
    auto query_set_view = raft::make_device_matrix_view<const T, int64_t>(
      query_set.data_handle(), query_set.extent(0), query_set.extent(1));
    auto host_query_set =
      raft::make_host_matrix<T, int64_t>(query_set.extent(0), query_set.extent(1));
#pragma omp parallel for
    for (auto i = start_idx; i < end_idx; i++) {
      auto pt_id = order[i];
      std::copy(host_dataset_view.data_handle() + pt_id * host_dataset_view.extent(1),
                host_dataset_view.data_handle() + (pt_id + 1) * host_dataset_view.extent(1),
                host_query_set.data_handle() + (i - start_idx) * host_dataset_view.extent(1));
    }
    raft::copy(query_set.data_handle(),
               host_query_set.data_handle(),
               query_set.size(),
               raft::resource::get_cuda_stream(res));

    // initialize brute force index
    auto bf_index = cuvs::neighbors::brute_force::build(
      res, cuvs::neighbors::brute_force::index_params{}, query_set_view);

    // search for nearest neighbors
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, num_pts, neighbor_size);
    auto distances = raft::make_device_matrix<float, int64_t>(res, num_pts, neighbor_size);
    cuvs::neighbors::brute_force::search(res,
                                         cuvs::neighbors::brute_force::search_params{},
                                         bf_index,
                                         query_set_view,
                                         neighbors.view(),
                                         distances.view());

    auto host_neighbors_full =
      raft::make_host_matrix<int64_t, int64_t>(neighbors.extent(0), neighbors.extent(1));
    raft::copy(host_neighbors_full.data_handle(),
               neighbors.data_handle(),
               neighbors.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    // Need to slice the first column of the neighbors matrix because it's the query point itself
    auto host_neighbors =
      raft::make_host_matrix<int64_t, int64_t>(neighbors.extent(0), neighbors.extent(1) - 1);
#pragma omp parallel for
    for (int64_t i = 0; i < host_neighbors.extent(0); i++) {
      std::copy(host_neighbors_full.data_handle() + i * host_neighbors_full.extent(1) + 1,
                host_neighbors_full.data_handle() + (i + 1) * host_neighbors_full.extent(1),
                host_neighbors.data_handle() + i * host_neighbors.extent(1));
    }

    // add points to the HNSW index upper layers
#pragma omp parallel for
    for (auto i = start_idx; i < end_idx; i++) {
      auto pt_id       = order[i];
      auto internal_id = hnsw_internal_ids[pt_id];
      auto ll_cur      = appr_algo->get_linklist(internal_id, pt_level);
      appr_algo->setListCount(ll_cur, host_neighbors.extent(1));
      auto* data     = (uint32_t*)(ll_cur + 1);
      auto neighbors = host_neighbors.data_handle() + (i - start_idx) * host_neighbors.extent(1);
      for (auto j = 0; j < host_neighbors.extent(1); j++) {
        auto neighbor_id          = order[neighbors[j] + start_idx];
        auto neighbor_internal_id = hnsw_internal_ids[neighbor_id];
        data[j]                   = neighbor_internal_id;
      }
    }
  }

  // move cagra graph to host
  auto graph = cagra_index.graph();
  auto host_graph =
    raft::make_host_matrix<uint32_t, int64_t, raft::row_major>(graph.extent(0), graph.extent(1));
  raft::copy(host_graph.data_handle(),
             graph.data_handle(),
             graph.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

// copy cagra graph to hnswlib base layer
#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(host_graph.extent(0)); ++i) {
    auto ll_i = appr_algo->get_linklist0(hnsw_internal_ids[i]);
    appr_algo->setListCount(ll_i, host_graph.extent(1));
    auto* data = (uint32_t*)(ll_i + 1);
    for (size_t j = 0; j < static_cast<size_t>(host_graph.extent(1)); ++j) {
      data[j] = hnsw_internal_ids[host_graph(i, j)];
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
    return from_cagra<T, HnswHierarchy::NONE>(res, params, cagra_index);
  } else if (params.hierarchy == HnswHierarchy::CPU) {
    return from_cagra<T, HnswHierarchy::CPU>(res, params, cagra_index, dataset);
  } else if (params.hierarchy == HnswHierarchy::GPU) {
    // brute force index does not support uint8_t or int8_t
    if constexpr (std::is_same_v<T, float>) {
      return from_cagra<T, HnswHierarchy::GPU>(res, params, cagra_index, dataset);
    } else {
      RAFT_FAIL("Unsupported data type for GPU hierarchy");
    }
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
  auto num_threads           = params.num_threads == 0 ? std::thread::hardware_concurrency()
                                                       : static_cast<size_t>(params.num_threads);

  hnswlib_index->resizeIndex(current_element_count + new_element_count);
  ParallelFor(current_element_count,
              current_element_count + new_element_count,
              num_threads,
              [&](size_t i, size_t threadId) {
                hnswlib_index->addPoint(
                  (void*)(additional_dataset.data_handle() +
                          (i - current_element_count) * additional_dataset.extent(1)),
                  i);
              });
}

template <typename T>
void get_search_knn_results(hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type> const* idx,
                            const T* query,
                            int k,
                            uint64_t* indices,
                            float* distances)
{
  // try {
  auto result = idx->searchKnn(query, k);
  // } catch (std::exception& e) {
  //   std::cout << "Caught exception: " << e.what() << std::endl;
  // }
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
