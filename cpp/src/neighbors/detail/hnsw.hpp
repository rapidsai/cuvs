/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../../core/omp_wrapper.hpp"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/util/file_io.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/util/cudart_utils.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <new>
#include <random>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace cuvs::neighbors::hnsw::detail {

// This is needed as hnswlib hardcodes the distance type to float
// or int32_t in certain places. However, we can solve uint8 or int8
// natively with the patch cuVS applies. We could potentially remove
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
struct hnsw_dist_t<half> {
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
    if (metric == cuvs::distance::DistanceType::InnerProduct) {
      space_ = std::make_unique<hnswlib::InnerProductSpace<T, typename hnsw_dist_t<T>::type>>(dim);
    } else if (metric == cuvs::distance::DistanceType::L2Expanded) {
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        space_ = std::make_unique<hnswlib::L2Space<T, typename hnsw_dist_t<T>::type>>(dim);
      } else if constexpr (std::is_same_v<T, std::int8_t> or std::is_same_v<T, std::uint8_t>) {
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
  void set_ef(int ef) const override
  {
    ensure_loaded();
    appr_alg_->ef_ = ef;
  }

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

  /**
  @brief Set file descriptor for disk-backed index
   */
  void set_file_descriptor(cuvs::util::file_descriptor&& fd) { hnsw_fd_.emplace(std::move(fd)); }

  /**
  @brief Get file descriptor
   */
  auto file_descriptor() const -> const std::optional<cuvs::util::file_descriptor>&
  {
    return hnsw_fd_;
  }

  /**
  @brief Get file path for disk-backed index
   */
  std::string file_path() const override
  {
    if (hnsw_fd_.has_value() && hnsw_fd_->is_valid()) { return hnsw_fd_->get_path(); }
    return "";
  }

  /**
  @brief Ensure the index is loaded into memory.
         If the index is disk-backed and not yet loaded, this will load it from the file.
   */
  void ensure_loaded() const
  {
    if (appr_alg_ != nullptr) { return; }  // Already loaded

    // Check if we have a file descriptor to load from
    if (!hnsw_fd_.has_value() || !hnsw_fd_->is_valid()) {
      RAFT_FAIL("Cannot load HNSW index: no file descriptor available and index not in memory");
    }

    std::string filepath = hnsw_fd_->get_path();
    RAFT_EXPECTS(!filepath.empty(), "Cannot load HNSW index: file path is empty");
    RAFT_EXPECTS(std::filesystem::exists(filepath),
                 "Cannot load HNSW index: file does not exist: %s",
                 filepath.c_str());

    RAFT_LOG_INFO("Loading HNSW index from disk: %s", filepath.c_str());

    try {
      appr_alg_ = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
        space_.get(), filepath);
      if (this->hierarchy() == HnswHierarchy::NONE) { appr_alg_->base_layer_only = true; }
    } catch (const std::bad_alloc& e) {
      RAFT_FAIL(
        "Failed to load HNSW index from '%s': insufficient host memory. "
        "The index is too large to fit in available RAM. "
        "Consider using a machine with more memory or reducing the dataset size.",
        filepath.c_str());
    }
  }

 private:
  mutable std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::unique_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;
  std::optional<cuvs::util::file_descriptor> hnsw_fd_;
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
    raft::copy_matrix(host_dataset.data_handle(),
                      host_dataset.extent(1),
                      cagra_dataset.data_handle(),
                      cagra_dataset.stride(0),
                      host_dataset.extent(1),
                      cagra_dataset.extent(0),
                      raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    host_dataset_view = host_dataset.view();
  }
  // build upper layers of hnsw index
  int dim         = host_dataset_view.extent(1);
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, cagra_index.metric(), hierarchy);
  auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(),
    host_dataset_view.extent(0),
    (cagra_index.graph().extent(1) + 1) / 2,
    params.ef_construction);
  appr_algo->base_layer_init = false;  // tell hnswlib to build upper layers only
  [[maybe_unused]] auto num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;
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
    raft::copy(res, host_graph.view(), host_graph_view);
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
  // FIXME: choose better heuristic
  bool use_nn_decent = neighbors.size() < 1e7;
  if (use_nn_decent) {
    nn_descent::index_params nn_params;
    nn_params.graph_degree              = neighbors.extent(1);
    nn_params.intermediate_graph_degree = neighbors.extent(1) * 2;
    nn_params.metric                    = metric;
    nn_params.return_distances          = false;
    auto nn_index                       = nn_descent::build(res, nn_params, dataset, neighbors);
  } else {
    // TODO: choose parameters to minimize memory consumption
    cagra::graph_build_params::ivf_pq_params ivfpq_params(dataset.extents(), metric);
    cagra::build_knn_graph(res, dataset, neighbors, ivfpq_params);
  }
}

template <typename T, typename IdxT>
void serialize_to_hnswlib_from_disk(raft::resources const& res,
                                    std::ostream& os_raw,
                                    const cuvs::neighbors::hnsw::index_params& params,
                                    const cuvs::neighbors::cagra::index<T, IdxT>& index_)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");

  auto start_time = std::chrono::system_clock::now();

  cuvs::util::buffered_ofstream os(&os_raw, 1 << 20 /*1MB*/);

  RAFT_EXPECTS(index_.dataset_fd().has_value() && index_.graph_fd().has_value(),
               "Function only implements serialization from disk.");
  RAFT_EXPECTS(params.hierarchy != HnswHierarchy::CPU,
               "Disk2disk serialization not supported for CPU hierarchy.");

  auto n_rows           = index_.size();
  auto dim              = index_.dim();
  auto graph_degree_int = static_cast<int>(index_.graph_degree());
  RAFT_LOG_INFO("Saving CAGRA index to hnswlib format, size %zu, dim %zu, graph_degree %zu",
                static_cast<size_t>(n_rows),
                static_cast<size_t>(dim),
                static_cast<size_t>(graph_degree_int));

  // Get file descriptors from index
  const auto& graph_fd_opt   = index_.graph_fd();
  const auto& dataset_fd_opt = index_.dataset_fd();
  const auto& mapping_fd_opt = index_.mapping_fd();

  RAFT_EXPECTS(graph_fd_opt.has_value() && graph_fd_opt->is_valid(),
               "Graph file descriptor is not available");
  RAFT_EXPECTS(dataset_fd_opt.has_value() && dataset_fd_opt->is_valid(),
               "Dataset file descriptor is not available");
  RAFT_EXPECTS(mapping_fd_opt.has_value() && mapping_fd_opt->is_valid(),
               "Mapping file descriptor is not available");

  // Get file paths from file descriptors
  std::string graph_path   = graph_fd_opt->get_path();
  std::string dataset_path = dataset_fd_opt->get_path();
  std::string mapping_path = mapping_fd_opt->get_path();

  RAFT_EXPECTS(!graph_path.empty(), "Unable to get path from graph file descriptor");
  RAFT_EXPECTS(!dataset_path.empty(), "Unable to get path from dataset file descriptor");
  RAFT_EXPECTS(!mapping_path.empty(), "Unable to get path from mapping file descriptor");

  int graph_fd   = graph_fd_opt->get();
  int dataset_fd = dataset_fd_opt->get();
  int label_fd   = mapping_fd_opt->get();

  // Read headers from files to get dimensions
  size_t graph_header_size = 0;
  size_t graph_n_rows      = 0;
  size_t graph_n_cols      = 0;
  {
    std::ifstream graph_stream(graph_path, std::ios::binary);
    RAFT_EXPECTS(graph_stream.good(), "Failed to open graph file: %s", graph_path.c_str());

    auto header       = raft::detail::numpy_serializer::read_header(graph_stream);
    graph_header_size = static_cast<size_t>(graph_stream.tellg());
    RAFT_EXPECTS(
      header.shape.size() == 2, "Graph file should be 2D, got %zu dimensions", header.shape.size());

    graph_n_rows = header.shape[0];
    graph_n_cols = header.shape[1];
    RAFT_LOG_DEBUG("Graph file: %zu x %zu, header size: %zu bytes",
                   graph_n_rows,
                   graph_n_cols,
                   graph_header_size);
  }

  size_t dataset_header_size = 0;
  size_t dataset_n_rows      = 0;
  size_t dataset_n_cols      = 0;
  {
    std::ifstream dataset_stream(dataset_path, std::ios::binary);
    RAFT_EXPECTS(dataset_stream.good(), "Failed to open dataset file: %s", dataset_path.c_str());

    auto header         = raft::detail::numpy_serializer::read_header(dataset_stream);
    dataset_header_size = static_cast<size_t>(dataset_stream.tellg());
    RAFT_EXPECTS(header.shape.size() == 2,
                 "Dataset file should be 2D, got %zu dimensions",
                 header.shape.size());

    dataset_n_rows = header.shape[0];
    dataset_n_cols = header.shape[1];
    RAFT_LOG_DEBUG("Dataset file: %zu x %zu, header size: %zu bytes",
                   dataset_n_rows,
                   dataset_n_cols,
                   dataset_header_size);
  }

  size_t label_header_size = 0;
  size_t label_n_elements  = 0;
  {
    std::ifstream mapping_stream(mapping_path, std::ios::binary);
    RAFT_EXPECTS(mapping_stream.good(), "Failed to open mapping file: %s", mapping_path.c_str());

    auto header       = raft::detail::numpy_serializer::read_header(mapping_stream);
    label_header_size = static_cast<size_t>(mapping_stream.tellg());
    RAFT_EXPECTS(header.shape.size() == 1,
                 "Mapping file should be 1D, got %zu dimensions",
                 header.shape.size());

    label_n_elements = header.shape[0];
    RAFT_LOG_DEBUG(
      "Mapping file: %zu elements, header size: %zu bytes", label_n_elements, label_header_size);
  }

  // Verify consistency
  RAFT_EXPECTS(graph_n_rows == static_cast<size_t>(n_rows),
               "Graph rows (%zu) != index size (%zu)",
               graph_n_rows,
               static_cast<size_t>(n_rows));
  RAFT_EXPECTS(dataset_n_rows == static_cast<size_t>(n_rows),
               "Dataset rows (%zu) != index size (%zu)",
               dataset_n_rows,
               static_cast<size_t>(n_rows));
  RAFT_EXPECTS(label_n_elements == static_cast<size_t>(n_rows),
               "Label elements (%zu) != index size (%zu)",
               label_n_elements,
               static_cast<size_t>(n_rows));
  RAFT_EXPECTS(graph_n_cols == static_cast<size_t>(graph_degree_int),
               "Graph cols (%zu) != graph degree (%d)",
               graph_n_cols,
               graph_degree_int);
  RAFT_EXPECTS(dataset_n_cols == static_cast<size_t>(dim),
               "Dataset cols (%zu) != dimensions (%zu)",
               dataset_n_cols,
               static_cast<size_t>(dim));

  const size_t row_size_bytes =
    graph_degree_int * sizeof(IdxT) + dim * sizeof(T) + sizeof(uint32_t);
  const size_t target_batch_bytes = 64 * 1024 * 1024;
  const size_t batch_size         = std::max<size_t>(1, target_batch_bytes / row_size_bytes);

  RAFT_LOG_DEBUG("Using batch size %zu rows (~%.2f MiB/batch)",
                 batch_size,
                 (batch_size * row_size_bytes) / (1024.0 * 1024.0));

  // Allocate buffers for batched reading
  auto graph_buffer   = raft::make_host_matrix<IdxT, int64_t>(batch_size, graph_degree_int);
  auto dataset_buffer = raft::make_host_matrix<T, int64_t>(batch_size, dim);
  auto label_buffer   = raft::make_host_vector<uint32_t, int64_t>(batch_size);

  RAFT_LOG_DEBUG("Allocated buffers: graph[%ld,%d], dataset[%ld,%ld], labels[%ld]",
                 graph_buffer.extent(0),
                 graph_degree_int,
                 dataset_buffer.extent(0),
                 dataset_buffer.extent(1),
                 label_buffer.extent(0));

  // initialize dummy HNSW index to retrieve constants
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, index_.metric(), params.hierarchy);

  int odd_graph_degree = graph_degree_int % 2;
  auto appr_algo       = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(), 1, (graph_degree_int + 1) / 2, params.ef_construction);

  bool create_hierarchy = params.hierarchy != HnswHierarchy::NONE;

  // create hierarchy order
  // sort the points by levels
  // roll dice & build histogram
  std::vector<size_t> hist;
  std::vector<size_t> order(n_rows);
  std::vector<size_t> order_bw(n_rows);
  std::vector<int> levels(n_rows);
  std::vector<size_t> offsets;

  if (create_hierarchy) {
    RAFT_LOG_INFO("Sort points by levels");
    for (int64_t i = 0; i < n_rows; i++) {
      auto pt_level = appr_algo->getRandomLevel(appr_algo->mult_);
      while (pt_level >= static_cast<int32_t>(hist.size()))
        hist.push_back(0);
      hist[pt_level]++;
      levels[i] = pt_level;
    }

    // accumulate
    offsets.resize(hist.size() + 1, 0);
    for (size_t i = 0; i < hist.size() - 1; i++) {
      offsets[i + 1] = offsets[i] + hist[i];
      RAFT_LOG_INFO("Level %zu : %zu", i + 1, size_t(n_rows) - offsets[i + 1]);
    }

    // fw/bw indices
    for (int64_t i = 0; i < n_rows; i++) {
      auto pt_level              = levels[i];
      order_bw[i]                = offsets[pt_level];
      order[offsets[pt_level]++] = i;
    }
  }

  // set last point of the highest level as the entry point
  appr_algo->enterpoint_node_ = create_hierarchy ? order.back() : n_rows / 2;
  appr_algo->maxlevel_        = create_hierarchy ? hist.size() - 1 : 1;

  // write header information
  RAFT_LOG_DEBUG("Writing HNSW header: offsetLevel0=%zu, n_rows=%zu, size_data_per_element=%zu",
                 appr_algo->offsetLevel0_,
                 static_cast<size_t>(n_rows),
                 appr_algo->size_data_per_element_);
  RAFT_LOG_DEBUG("  maxlevel=%d, enterpoint=%d, maxM=%zu, maxM0=%zu, M=%zu",
                 appr_algo->maxlevel_,
                 appr_algo->enterpoint_node_,
                 appr_algo->maxM_,
                 appr_algo->maxM0_,
                 appr_algo->M_);

  // offset_level_0
  os.write(reinterpret_cast<char*>(&appr_algo->offsetLevel0_), sizeof(std::size_t));
  // 8 max_element - override with n_rows
  size_t num_elements = (size_t)n_rows;
  os.write(reinterpret_cast<char*>(&num_elements), sizeof(std::size_t));
  // 16 curr_element_count - override with n_rows
  os.write(reinterpret_cast<char*>(&num_elements), sizeof(std::size_t));
  // 24 size_data_per_element
  os.write(reinterpret_cast<char*>(&appr_algo->size_data_per_element_), sizeof(std::size_t));
  // 32 label_offset
  os.write(reinterpret_cast<char*>(&appr_algo->label_offset_), sizeof(std::size_t));
  // 40 offset_data
  os.write(reinterpret_cast<char*>(&appr_algo->offsetData_), sizeof(std::size_t));
  // 48 maxlevel
  os.write(reinterpret_cast<char*>(&appr_algo->maxlevel_), sizeof(int));
  // 52 enterpoint_node
  os.write(reinterpret_cast<char*>(&appr_algo->enterpoint_node_), sizeof(int));
  // 56 maxM
  os.write(reinterpret_cast<char*>(&appr_algo->maxM_), sizeof(std::size_t));
  // 64 maxM0
  os.write(reinterpret_cast<char*>(&appr_algo->maxM0_), sizeof(std::size_t));
  // 72 M
  os.write(reinterpret_cast<char*>(&appr_algo->M_), sizeof(std::size_t));
  // 80 mult
  os.write(reinterpret_cast<char*>(&appr_algo->mult_), sizeof(double));
  // 88 ef_construction
  os.write(reinterpret_cast<char*>(&appr_algo->ef_construction_), sizeof(std::size_t));

  // host queries
  auto host_query_set =
    raft::make_host_matrix<T, int64_t>(create_hierarchy ? n_rows - hist[0] : 0, dim);

  int64_t d_report_offset    = n_rows / 10;  // Report progress in 10% steps.
  int64_t next_report_offset = d_report_offset;
  auto start_clock           = std::chrono::system_clock::now();

  RAFT_EXPECTS(appr_algo->size_data_per_element_ ==
                 dim * sizeof(T) + appr_algo->maxM0_ * sizeof(IdxT) + sizeof(int) + sizeof(size_t),
               "Size data per element mismatch");

  RAFT_LOG_INFO("Writing base level");
  size_t bytes_written = 0;
  float GiB            = 1 << 30;
  IdxT zero            = 0;
  RAFT_EXPECTS(appr_algo->size_data_per_element_ ==
                 dim * sizeof(T) + appr_algo->maxM0_ * sizeof(IdxT) + sizeof(int) + sizeof(size_t),
               "Size data per element mismatch");

  // Helper lambda for parallel reading of batches
  auto read_batch = [&](int64_t start_row, int64_t rows_to_read) {
    const size_t graph_bytes   = rows_to_read * graph_degree_int * sizeof(IdxT);
    const size_t dataset_bytes = rows_to_read * dim * sizeof(T);
    const size_t label_bytes   = rows_to_read * sizeof(uint32_t);

    const off_t graph_offset   = graph_header_size + start_row * graph_degree_int * sizeof(IdxT);
    const off_t dataset_offset = dataset_header_size + start_row * dim * sizeof(T);
    const off_t label_offset   = label_header_size + start_row * sizeof(uint32_t);

    RAFT_LOG_DEBUG("Reading batch: row=%ld, rows=%ld", start_row, rows_to_read);
    RAFT_LOG_DEBUG(
      "  graph: offset=%zu, bytes=%zu", static_cast<size_t>(graph_offset), graph_bytes);
    RAFT_LOG_DEBUG(
      "  dataset: offset=%zu, bytes=%zu", static_cast<size_t>(dataset_offset), dataset_bytes);
    RAFT_LOG_DEBUG(
      "  label: offset=%zu, bytes=%zu", static_cast<size_t>(label_offset), label_bytes);

#pragma omp parallel sections num_threads(3)
    {
#pragma omp section
      {
        ssize_t bytes_read = pread(graph_fd, graph_buffer.data_handle(), graph_bytes, graph_offset);
        RAFT_EXPECTS(bytes_read == static_cast<ssize_t>(graph_bytes),
                     "Failed to read graph data: expected %zu, got %zd",
                     graph_bytes,
                     bytes_read);
      }
#pragma omp section
      {
        ssize_t bytes_read =
          pread(dataset_fd, dataset_buffer.data_handle(), dataset_bytes, dataset_offset);
        RAFT_EXPECTS(bytes_read == static_cast<ssize_t>(dataset_bytes),
                     "Failed to read dataset data: expected %zu, got %zd",
                     dataset_bytes,
                     bytes_read);
      }
#pragma omp section
      {
        ssize_t bytes_read = pread(label_fd, label_buffer.data_handle(), label_bytes, label_offset);
        RAFT_EXPECTS(bytes_read == static_cast<ssize_t>(label_bytes),
                     "Failed to read label data: expected %zu, got %zd",
                     label_bytes,
                     bytes_read);
      }
    }

    // Log first few values from first batch for debugging
    if (start_row == 0 && rows_to_read > 0) {
      RAFT_LOG_DEBUG("First graph row: [%u, %u, %u, ...]",
                     static_cast<unsigned int>(graph_buffer(0, 0)),
                     graph_degree_int > 1 ? static_cast<unsigned int>(graph_buffer(0, 1)) : 0,
                     graph_degree_int > 2 ? static_cast<unsigned int>(graph_buffer(0, 2)) : 0);
      RAFT_LOG_DEBUG("First dataset row: [%f, %f, %f, ...]",
                     static_cast<float>(dataset_buffer(0, 0)),
                     dim > 1 ? static_cast<float>(dataset_buffer(0, 1)) : 0.0f,
                     dim > 2 ? static_cast<float>(dataset_buffer(0, 2)) : 0.0f);
      RAFT_LOG_DEBUG("First labels: [%u, %u, %u, ...]",
                     static_cast<unsigned int>(label_buffer(0)),
                     rows_to_read > 1 ? static_cast<unsigned int>(label_buffer(1)) : 0,
                     rows_to_read > 2 ? static_cast<unsigned int>(label_buffer(2)) : 0);
    }
  };

  for (int64_t batch_start = 0; batch_start < n_rows; batch_start += batch_size) {
    const int64_t current_batch_size = std::min<int64_t>(batch_size, n_rows - batch_start);

    RAFT_LOG_DEBUG("Reading batch: start=%ld, size=%ld (batch_size=%zu)",
                   batch_start,
                   current_batch_size,
                   batch_size);
    read_batch(batch_start, current_batch_size);

    for (int64_t batch_idx = 0; batch_idx < current_batch_size; batch_idx++) {
      const int64_t i = batch_start + batch_idx;

      os.write(reinterpret_cast<char*>(&graph_degree_int), sizeof(int));

      const IdxT* graph_row = &graph_buffer(batch_idx, 0);
      os.write(reinterpret_cast<const char*>(graph_row), sizeof(IdxT) * graph_degree_int);

      if (odd_graph_degree) {
        RAFT_EXPECTS(odd_graph_degree == static_cast<int>(appr_algo->maxM0_) - graph_degree_int,
                     "Odd graph degree mismatch");
        os.write(reinterpret_cast<char*>(&zero), sizeof(IdxT));
      }

      const T* data_row = &dataset_buffer(batch_idx, 0);
      os.write(reinterpret_cast<const char*>(data_row), sizeof(T) * dim);

      if (create_hierarchy && levels[i] > 0) {
        // position in query: order_bw[i]-hist[0]
        std::copy(data_row,
                  data_row + dim,
                  reinterpret_cast<char*>(&host_query_set(order_bw[i] - hist[0], 0)));
      }

      // assign original label
      auto label = static_cast<size_t>(label_buffer(batch_idx));
      os.write(reinterpret_cast<char*>(&label), sizeof(std::size_t));

      bytes_written += appr_algo->size_data_per_element_;

      const auto end_clock = std::chrono::system_clock::now();
      // if (!os.good()) { RAFT_FAIL("Error writing HNSW file, row %zu", i); }
      if (i > next_report_offset) {
        next_report_offset += d_report_offset;
        const auto time =
          std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
          1e-6;
        float throughput      = bytes_written / GiB / time;
        float rows_throughput = i / time;
        float ETA             = (n_rows - i) / rows_throughput;
        RAFT_LOG_INFO(
          "# Writing rows %12lu / %12lu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f "
          "GiB\r",
          i,
          n_rows,
          i / static_cast<double>(n_rows) * 100,
          throughput,
          int(ETA / 60),
          std::fmod(ETA, 60.0f),
          bytes_written / GiB);
      }
    }
  }

  RAFT_LOG_DEBUG("Completed writing %ld base level rows", n_rows);

  // trigger knn builds for all levels
  std::vector<raft::host_matrix<IdxT, int64_t>> host_neighbors;
  if (create_hierarchy) {
    for (size_t pt_level = 1; pt_level < hist.size(); pt_level++) {
      auto num_pts       = n_rows - offsets[pt_level - 1];
      auto neighbor_size = num_pts > appr_algo->M_ ? appr_algo->M_ : num_pts - 1;
      host_neighbors.emplace_back(raft::make_host_matrix<IdxT, int64_t>(num_pts, neighbor_size));
    }
    for (size_t pt_level = 1; pt_level < hist.size(); pt_level++) {
      RAFT_LOG_INFO("Compute hierarchy neighbors level %zu", pt_level);
      auto removed_rows = offsets[pt_level - 1] - offsets[0];
      raft::host_matrix_view<T, int64_t, raft::row_major> sub_query_view(
        host_query_set.data_handle() + removed_rows * dim,
        host_query_set.extent(0) - removed_rows,
        dim);
      auto neighbor_view = host_neighbors[pt_level - 1].view();
      all_neighbors_graph(
        res, raft::make_const_mdspan(sub_query_view), neighbor_view, index_.metric());
    }
  }

  if (create_hierarchy) {
    RAFT_LOG_INFO("Assemble hierarchy linklists");
    next_report_offset = d_report_offset;
  }
  bytes_written = 0;
  start_clock   = std::chrono::system_clock::now();

  for (int64_t i = 0; i < n_rows; i++) {
    size_t cur_level = create_hierarchy ? levels[i] : 0;
    unsigned int linkListSize =
      create_hierarchy && cur_level > 0 ? appr_algo->size_links_per_element_ * cur_level : 0;
    os.write(reinterpret_cast<char*>(&linkListSize), sizeof(int));
    bytes_written += sizeof(int);
    if (linkListSize) {
      for (size_t pt_level = 1; pt_level <= cur_level; pt_level++) {
        auto neighbor_view = host_neighbors[pt_level - 1].view();
        auto my_row        = order_bw[i] - offsets[pt_level - 1];

        IdxT* neighbors     = &neighbor_view(my_row, 0);
        unsigned int extent = neighbor_view.extent(1);
        os.write(reinterpret_cast<char*>(&extent), sizeof(int));
        for (unsigned int j = 0; j < extent; j++) {
          const IdxT converted = order[neighbors[j] + offsets[pt_level - 1]];
          os.write(reinterpret_cast<const char*>(&converted), sizeof(IdxT));
        }
        auto remainder = appr_algo->M_ - neighbor_view.extent(1);
        for (size_t j = 0; j < remainder; j++) {
          os.write(reinterpret_cast<char*>(&zero), sizeof(IdxT));
        }
        bytes_written += (neighbor_view.extent(1) + remainder) * sizeof(IdxT) + sizeof(int);
        RAFT_EXPECTS(appr_algo->size_links_per_element_ ==
                       (neighbor_view.extent(1) + remainder) * sizeof(IdxT) + sizeof(int),
                     "Size links per element mismatch");
      }
    }

    const auto end_clock = std::chrono::system_clock::now();
    if (i > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      float throughput      = bytes_written / GiB / time;
      float rows_throughput = i / time;
      float ETA             = (n_rows - i) / rows_throughput;
      RAFT_LOG_INFO(
        "# Writing rows %12lu / %12lu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f GiB\r",
        i,
        n_rows,
        i / static_cast<double>(n_rows) * 100,
        throughput,
        int(ETA / 60),
        std::fmod(ETA, 60.0f),
        bytes_written / GiB);
    }
  }

  // Flush buffered output and check data was written
  os.flush();
  os_raw.flush();
  auto final_pos = os_raw.tellp();
  RAFT_LOG_DEBUG("HNSW file size: %ld bytes", static_cast<int64_t>(final_pos));
  if (!os_raw.good()) { RAFT_LOG_WARN("Output stream is not in good state after serialization"); }

  auto end_time = std::chrono::system_clock::now();
  auto elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  RAFT_LOG_INFO("HNSW serialization from disk complete in %ld ms", elapsed_time);
}

template <typename T, typename IdxT>
std::pair<cuvs::util::file_descriptor, size_t> remap_disk_graph_to_original_ids(
  const cuvs::neighbors::cagra::index<T, IdxT>& index_, const std::string& output_path)
{
  auto total_start           = std::chrono::high_resolution_clock::now();
  const auto& graph_fd_opt   = index_.graph_fd();
  const auto& mapping_fd_opt = index_.mapping_fd();

  RAFT_EXPECTS(graph_fd_opt.has_value() && graph_fd_opt->is_valid(),
               "Graph file descriptor is not available");
  RAFT_EXPECTS(mapping_fd_opt.has_value() && mapping_fd_opt->is_valid(),
               "Mapping file descriptor is not available");

  const auto graph_path   = graph_fd_opt->get_path();
  const auto mapping_path = mapping_fd_opt->get_path();
  RAFT_EXPECTS(!graph_path.empty(), "Unable to get path from graph file descriptor");
  RAFT_EXPECTS(!mapping_path.empty(), "Unable to get path from mapping file descriptor");

  std::ifstream graph_stream(graph_path, std::ios::binary);
  RAFT_EXPECTS(graph_stream.good(), "Failed to open graph file: %s", graph_path.c_str());
  auto graph_header      = raft::detail::numpy_serializer::read_header(graph_stream);
  auto graph_header_size = static_cast<size_t>(graph_stream.tellg());
  RAFT_EXPECTS(graph_header.shape.size() == 2,
               "Graph file should be 2D, got %zu dimensions",
               graph_header.shape.size());

  std::ifstream mapping_stream(mapping_path, std::ios::binary);
  RAFT_EXPECTS(mapping_stream.good(), "Failed to open mapping file: %s", mapping_path.c_str());
  auto mapping_header      = raft::detail::numpy_serializer::read_header(mapping_stream);
  auto mapping_header_size = static_cast<size_t>(mapping_stream.tellg());
  RAFT_EXPECTS(mapping_header.shape.size() == 1,
               "Mapping file should be 1D, got %zu dimensions",
               mapping_header.shape.size());

  const auto n_rows       = graph_header.shape[0];
  const auto graph_degree = graph_header.shape[1];
  RAFT_EXPECTS(mapping_header.shape[0] == n_rows,
               "Mapping size (%zu) must match graph rows (%zu)",
               mapping_header.shape[0],
               n_rows);

  auto mapping = raft::make_host_vector<IdxT, int64_t>(n_rows);
  cuvs::util::read_large_file(
    *mapping_fd_opt, mapping.data_handle(), n_rows * sizeof(IdxT), mapping_header_size);
  const auto* mapping_data = mapping.data_handle();

  auto reordered_rows       = raft::make_host_vector<IdxT, int64_t>(n_rows);
  auto* reordered_rows_data = reordered_rows.data_handle();
#pragma omp parallel for
  for (int64_t reordered_row = 0; reordered_row < static_cast<int64_t>(n_rows); ++reordered_row) {
    reordered_rows_data[mapping_data[reordered_row]] = static_cast<IdxT>(reordered_row);
  }

  auto [output_fd, output_header_size] =
    cuvs::util::create_numpy_file<IdxT>(output_path, {n_rows, graph_degree});

  // Target batch size and coalescing thresholds. Larger batches require more RAM.
  // Tune these values if you see performance issues.
  const size_t target_batch_bytes       = 64 * 1024 * 1024;
  const size_t max_coalesced_gap_bytes  = 128 * 1024;
  const size_t max_coalesced_span_bytes = 8 * 1024 * 1024;
  const size_t row_bytes                = graph_degree * sizeof(IdxT);
  const size_t batch_size               = std::max<size_t>(1, target_batch_bytes / row_bytes);
  const size_t max_coalesced_gap_rows   = std::max<size_t>(1, max_coalesced_gap_bytes / row_bytes);
  const size_t max_coalesced_span_rows  = std::max<size_t>(1, max_coalesced_span_bytes / row_bytes);

  // Limit the number of read threads. Larger values require more RAM. 4-8 should be able to
  // saturate most NVMe disks. Tune this value if you see I/O performance issues.
  const int read_parallelism = std::min(cuvs::core::omp::get_max_threads(), 8);
  auto output_batch =
    raft::make_host_matrix<IdxT, int64_t>(batch_size, static_cast<int64_t>(graph_degree));
  auto span_buffer = raft::make_host_vector<IdxT, int64_t>(static_cast<int64_t>(read_parallelism) *
                                                           max_coalesced_span_rows * graph_degree);
  std::vector<size_t> read_order(batch_size);
  auto* output_batch_data = output_batch.data_handle();
  auto* span_buffer_data  = span_buffer.data_handle();

  // Bucket sort read_order by coarse disk region if buckets are not too sparse.
  const size_t bucket_size_rows = max_coalesced_span_rows;
  const size_t num_buckets      = (n_rows + bucket_size_rows - 1) / bucket_size_rows;
  const bool use_bucket_sort    = num_buckets > 0 && batch_size / num_buckets >= 4;
  std::vector<size_t> bucket_offsets(use_bucket_sort ? num_buckets + 1 : 0);
  std::vector<size_t> bucket_scatter(use_bucket_sort ? num_buckets : 0);

  struct span_t {
    size_t sorted_begin;
    size_t sorted_end;
    size_t first_reordered_row;
    size_t span_rows;
  };
  std::vector<span_t> spans;
  spans.reserve(batch_size);

  size_t total_read_bytes  = 0;
  size_t total_write_bytes = 0;
  size_t total_span_count  = 0;

  RAFT_LOG_INFO(
    "HNSW remap: n_rows=%zu degree=%zu batch_size=%zu max_gap=%zu rows max_span=%zu rows "
    "read_threads=%d",
    n_rows,
    graph_degree,
    batch_size,
    max_coalesced_gap_rows,
    max_coalesced_span_rows,
    read_parallelism);

  for (size_t original_batch_start = 0; original_batch_start < n_rows;
       original_batch_start += batch_size) {
    const auto current_batch_size = std::min(batch_size, n_rows - original_batch_start);

    const auto key_less = [&](size_t lhs, size_t rhs) {
      return reordered_rows_data[original_batch_start + lhs] <
             reordered_rows_data[original_batch_start + rhs];
    };
    if (use_bucket_sort) {
      std::fill(bucket_offsets.begin(), bucket_offsets.end(), 0);
      for (size_t k = 0; k < current_batch_size; ++k) {
        const auto rr = reordered_rows_data[original_batch_start + k];
        bucket_offsets[rr / bucket_size_rows + 1]++;
      }
      for (size_t b = 1; b <= num_buckets; ++b) {
        bucket_offsets[b] += bucket_offsets[b - 1];
      }
      std::copy(
        bucket_offsets.begin(), bucket_offsets.begin() + num_buckets, bucket_scatter.begin());
      for (size_t k = 0; k < current_batch_size; ++k) {
        const auto rr = reordered_rows_data[original_batch_start + k];
        read_order[bucket_scatter[rr / bucket_size_rows]++] = k;
      }
      for (size_t b = 0; b < num_buckets; ++b) {
        const auto lo = bucket_offsets[b];
        const auto hi = bucket_offsets[b + 1];
        if (hi - lo > 1) { std::sort(read_order.begin() + lo, read_order.begin() + hi, key_less); }
      }
    } else {
      for (size_t k = 0; k < current_batch_size; ++k) {
        read_order[k] = k;
      }
      std::sort(read_order.begin(), read_order.begin() + current_batch_size, key_less);
    }

    // Precompute coalesced spans so reads and remap can run in parallel across spans.
    spans.clear();
    for (size_t sorted_pos = 0; sorted_pos < current_batch_size;) {
      const auto first_reordered_row =
        static_cast<size_t>(reordered_rows_data[original_batch_start + read_order[sorted_pos]]);
      size_t span_end_pos       = sorted_pos + 1;
      size_t last_reordered_row = first_reordered_row;
      while (span_end_pos < current_batch_size) {
        const auto next_reordered_row =
          static_cast<size_t>(reordered_rows_data[original_batch_start + read_order[span_end_pos]]);
        const auto gap_rows  = next_reordered_row - last_reordered_row;
        const auto span_rows = next_reordered_row - first_reordered_row + 1;
        if (gap_rows > max_coalesced_gap_rows || span_rows > max_coalesced_span_rows) { break; }
        last_reordered_row = next_reordered_row;
        ++span_end_pos;
      }
      spans.push_back({sorted_pos,
                       span_end_pos,
                       first_reordered_row,
                       last_reordered_row - first_reordered_row + 1});
      sorted_pos = span_end_pos;
    }

    size_t batch_read_bytes = 0;
#pragma omp parallel for num_threads(read_parallelism) reduction(+ : batch_read_bytes)
    for (int64_t span_idx = 0; span_idx < static_cast<int64_t>(spans.size()); ++span_idx) {
      const auto& s    = spans[span_idx];
      const int tid    = cuvs::core::omp::get_thread_num();
      IdxT* tid_buffer = span_buffer_data + tid * max_coalesced_span_rows * graph_degree;
      const auto input_offset =
        static_cast<uint64_t>(graph_header_size + s.first_reordered_row * row_bytes);
      const auto bytes_to_read = s.span_rows * row_bytes;
      cuvs::util::read_large_file(*graph_fd_opt, tid_buffer, bytes_to_read, input_offset);
      batch_read_bytes += bytes_to_read;

      for (size_t pos = s.sorted_begin; pos < s.sorted_end; ++pos) {
        const auto batch_idx      = read_order[pos];
        const auto original_row   = original_batch_start + batch_idx;
        const auto reordered_row  = static_cast<size_t>(reordered_rows_data[original_row]);
        const auto local_row      = reordered_row - s.first_reordered_row;
        const auto* graph_row_ptr = tid_buffer + local_row * graph_degree;
        auto* output_row_ptr      = output_batch_data + batch_idx * graph_degree;
        for (size_t neighbor_idx = 0; neighbor_idx < graph_degree; ++neighbor_idx) {
          output_row_ptr[neighbor_idx] = mapping_data[graph_row_ptr[neighbor_idx]];
        }
      }
    }

    const auto output_offset =
      static_cast<uint64_t>(output_header_size + original_batch_start * row_bytes);
    const auto batch_write_bytes = current_batch_size * row_bytes;
    cuvs::util::write_large_file(output_fd, output_batch_data, batch_write_bytes, output_offset);

    total_read_bytes += batch_read_bytes;
    total_write_bytes += batch_write_bytes;
    total_span_count += spans.size();
  }

  const auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::high_resolution_clock::now() - total_start)
                               .count();
  const double gib = static_cast<double>(1 << 30);
  RAFT_LOG_INFO(
    "HNSW remap: completed in %ld ms: read %.2f GiB across %zu spans (%.2fx amplification), "
    "wrote %.2f GiB",
    total_elapsed,
    total_read_bytes / gib,
    total_span_count,
    total_write_bytes > 0 ? static_cast<double>(total_read_bytes) / total_write_bytes : 0.0,
    total_write_bytes / gib);

  return {std::move(output_fd), output_header_size};
}

template <typename T, typename IdxT>
void serialize_to_hnswlib_with_original_dataset(
  raft::resources const& res,
  std::ostream& os_raw,
  const cuvs::neighbors::hnsw::index_params& params,
  const cuvs::neighbors::cagra::index<T, IdxT>& index_,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::serialize_original_order");

  auto start_time = std::chrono::system_clock::now();
  cuvs::util::buffered_ofstream os(&os_raw, 1 << 20 /*1MB*/);

  RAFT_EXPECTS(index_.graph_fd().has_value() && index_.mapping_fd().has_value(),
               "Function only implements serialization from disk-backed ACE graph.");
  RAFT_EXPECTS(params.hierarchy != HnswHierarchy::CPU,
               "Disk2disk serialization not supported for CPU hierarchy.");
  RAFT_EXPECTS(static_cast<size_t>(dataset.extent(0)) == static_cast<size_t>(index_.size()),
               "Dataset rows (%zu) must match index size (%zu)",
               static_cast<size_t>(dataset.extent(0)),
               static_cast<size_t>(index_.size()));
  RAFT_EXPECTS(static_cast<size_t>(dataset.extent(1)) == static_cast<size_t>(index_.dim()),
               "Dataset cols (%zu) must match index dimensions (%zu)",
               static_cast<size_t>(dataset.extent(1)),
               static_cast<size_t>(index_.dim()));

  const auto& graph_fd   = index_.graph_fd();
  std::string graph_path = graph_fd->get_path();
  RAFT_EXPECTS(!graph_path.empty(), "Unable to get path from graph file descriptor");
  const auto index_directory = std::filesystem::path(graph_path).parent_path().string();
  const auto remapped_graph_path =
    (std::filesystem::path(index_directory) / "cagra_graph_original_ids.npy").string();

  auto [remapped_graph_fd, graph_header_size] =
    remap_disk_graph_to_original_ids(index_, remapped_graph_path);

  auto n_rows           = index_.size();
  auto dim              = index_.dim();
  auto graph_degree_int = static_cast<int>(index_.graph_degree());

  RAFT_LOG_INFO(
    "Saving CAGRA index to hnswlib format with original dataset order, size %zu, dim %zu, "
    "graph_degree %zu",
    static_cast<size_t>(n_rows),
    static_cast<size_t>(dim),
    static_cast<size_t>(graph_degree_int));

  // Size the per-batch graph read buffer around a 64 MiB read target.
  const size_t target_batch_bytes = 64 * 1024 * 1024;
  const size_t batch_size =
    std::max<size_t>(1, target_batch_bytes / (graph_degree_int * sizeof(IdxT)));

  auto graph_buffer = raft::make_host_matrix<IdxT, int64_t>(batch_size, graph_degree_int);

  auto hnsw_index = std::make_unique<index_impl<T>>(dim, index_.metric(), params.hierarchy);

  int odd_graph_degree = graph_degree_int % 2;
  auto appr_algo       = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(), 1, (graph_degree_int + 1) / 2, params.ef_construction);

  bool create_hierarchy = params.hierarchy != HnswHierarchy::NONE;

  std::vector<size_t> hist;
  std::vector<size_t> order(n_rows);
  std::vector<size_t> order_bw(n_rows);
  std::vector<int> levels(n_rows);
  std::vector<size_t> offsets;

  if (create_hierarchy) {
    RAFT_LOG_INFO("Sort points by levels");
    for (int64_t i = 0; i < n_rows; i++) {
      auto pt_level = appr_algo->getRandomLevel(appr_algo->mult_);
      while (pt_level >= static_cast<int32_t>(hist.size()))
        hist.push_back(0);
      hist[pt_level]++;
      levels[i] = pt_level;
    }

    offsets.resize(hist.size() + 1, 0);
    for (size_t i = 0; i < hist.size() - 1; i++) {
      offsets[i + 1] = offsets[i] + hist[i];
      RAFT_LOG_INFO("Level %zu : %zu", i + 1, size_t(n_rows) - offsets[i + 1]);
    }

    for (int64_t i = 0; i < n_rows; i++) {
      auto pt_level              = levels[i];
      order_bw[i]                = offsets[pt_level];
      order[offsets[pt_level]++] = i;
    }
  }

  appr_algo->enterpoint_node_ = create_hierarchy ? order.back() : n_rows / 2;
  appr_algo->maxlevel_        = create_hierarchy ? hist.size() - 1 : 1;

  os.write(reinterpret_cast<char*>(&appr_algo->offsetLevel0_), sizeof(std::size_t));
  size_t num_elements = static_cast<size_t>(n_rows);
  os.write(reinterpret_cast<char*>(&num_elements), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&num_elements), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&appr_algo->size_data_per_element_), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&appr_algo->label_offset_), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&appr_algo->offsetData_), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&appr_algo->maxlevel_), sizeof(int));
  os.write(reinterpret_cast<char*>(&appr_algo->enterpoint_node_), sizeof(int));
  os.write(reinterpret_cast<char*>(&appr_algo->maxM_), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&appr_algo->maxM0_), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&appr_algo->M_), sizeof(std::size_t));
  os.write(reinterpret_cast<char*>(&appr_algo->mult_), sizeof(double));
  os.write(reinterpret_cast<char*>(&appr_algo->ef_construction_), sizeof(std::size_t));

  auto host_query_set =
    raft::make_host_matrix<T, int64_t>(create_hierarchy ? n_rows - hist[0] : 0, dim);

  int64_t d_report_offset    = n_rows / 10;
  int64_t next_report_offset = d_report_offset;
  auto start_clock           = std::chrono::system_clock::now();

  RAFT_EXPECTS(appr_algo->size_data_per_element_ ==
                 dim * sizeof(T) + appr_algo->maxM0_ * sizeof(IdxT) + sizeof(int) + sizeof(size_t),
               "Size data per element mismatch");

  RAFT_LOG_INFO("Writing base level");
  size_t bytes_written = 0;
  float GiB            = 1 << 30;
  IdxT zero            = 0;

  for (int64_t batch_start = 0; batch_start < n_rows; batch_start += batch_size) {
    const int64_t current_batch_size = std::min<int64_t>(batch_size, n_rows - batch_start);

    const size_t graph_bytes = current_batch_size * graph_degree_int * sizeof(IdxT);
    const off_t graph_offset = graph_header_size + batch_start * graph_degree_int * sizeof(IdxT);
    auto bytes_read =
      pread(remapped_graph_fd.get(), graph_buffer.data_handle(), graph_bytes, graph_offset);
    RAFT_EXPECTS(bytes_read == static_cast<ssize_t>(graph_bytes),
                 "Failed to read remapped graph data: expected %zu, got %zd",
                 graph_bytes,
                 bytes_read);

    for (int64_t batch_idx = 0; batch_idx < current_batch_size; batch_idx++) {
      const int64_t i = batch_start + batch_idx;

      os.write(reinterpret_cast<char*>(&graph_degree_int), sizeof(int));

      const IdxT* graph_row = &graph_buffer(batch_idx, 0);
      os.write(reinterpret_cast<const char*>(graph_row), sizeof(IdxT) * graph_degree_int);

      if (odd_graph_degree) {
        RAFT_EXPECTS(odd_graph_degree == static_cast<int>(appr_algo->maxM0_) - graph_degree_int,
                     "Odd graph degree mismatch");
        os.write(reinterpret_cast<char*>(&zero), sizeof(IdxT));
      }

      const T* data_row = &dataset(batch_start + batch_idx, 0);
      os.write(reinterpret_cast<const char*>(data_row), sizeof(T) * dim);

      if (create_hierarchy && levels[i] > 0) {
        std::copy(data_row, data_row + dim, &host_query_set(order_bw[i] - hist[0], 0));
      }

      auto label = static_cast<size_t>(i);
      os.write(reinterpret_cast<char*>(&label), sizeof(std::size_t));

      bytes_written += appr_algo->size_data_per_element_;

      const auto end_clock = std::chrono::system_clock::now();
      if (i > next_report_offset) {
        next_report_offset += d_report_offset;
        const auto time =
          std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
          1e-6;
        float throughput      = bytes_written / GiB / time;
        float rows_throughput = i / time;
        float ETA             = (n_rows - i) / rows_throughput;
        RAFT_LOG_INFO(
          "# Writing rows %12lu / %12lu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f "
          "GiB\r",
          i,
          n_rows,
          i / static_cast<double>(n_rows) * 100,
          throughput,
          int(ETA / 60),
          std::fmod(ETA, 60.0f),
          bytes_written / GiB);
      }
    }
  }

  RAFT_LOG_INFO("Writing upper layers");
  std::vector<std::optional<raft::host_matrix<uint32_t, int64_t>>> host_neighbors;
  host_neighbors.resize(hist.size());
  for (size_t pt_level = 1; create_hierarchy && pt_level < hist.size(); pt_level++) {
    common::nvtx::range<common::nvtx::domain::cuvs> level_scope("level %zu", pt_level);
    auto start_idx     = offsets[pt_level - 1];
    auto end_idx       = offsets[hist.size() - 1];
    auto num_pts       = end_idx - start_idx;
    auto neighbor_size = num_pts > appr_algo->M_ ? appr_algo->M_ : num_pts - 1;
    if (num_pts <= 1) {
      host_neighbors[pt_level - 1] = std::nullopt;
      continue;
    }

    auto view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      &host_query_set(start_idx - hist[0], 0), num_pts, dim);
    host_neighbors[pt_level - 1].emplace(
      raft::make_host_matrix<uint32_t, int64_t>(num_pts, neighbor_size));
    all_neighbors_graph(res, view, host_neighbors[pt_level - 1]->view(), index_.metric());
  }

  next_report_offset = d_report_offset;
  for (int64_t i = 0; i < n_rows; i++) {
    size_t cur_level = create_hierarchy ? levels[i] : 0;
    unsigned int linkListSize =
      create_hierarchy && cur_level > 0 ? appr_algo->size_links_per_element_ * cur_level : 0;
    os.write(reinterpret_cast<char*>(&linkListSize), sizeof(int));
    bytes_written += sizeof(int);
    if (linkListSize) {
      for (size_t pt_level = 1; pt_level <= cur_level; pt_level++) {
        unsigned int extent = 0;
        if (host_neighbors[pt_level - 1].has_value()) {
          auto neighbor_view = host_neighbors[pt_level - 1]->view();
          auto my_row        = order_bw[i] - offsets[pt_level - 1];
          IdxT* neighbors    = &neighbor_view(my_row, 0);
          extent             = neighbor_view.extent(1);
          os.write(reinterpret_cast<char*>(&extent), sizeof(int));
          for (unsigned int j = 0; j < extent; j++) {
            const IdxT converted = order[neighbors[j] + offsets[pt_level - 1]];
            os.write(reinterpret_cast<const char*>(&converted), sizeof(IdxT));
          }
        } else {
          os.write(reinterpret_cast<char*>(&extent), sizeof(int));
        }
        auto remainder = appr_algo->M_ - extent;
        for (size_t j = 0; j < remainder; j++) {
          os.write(reinterpret_cast<char*>(&zero), sizeof(IdxT));
        }
        bytes_written += (extent + remainder) * sizeof(IdxT) + sizeof(int);
        RAFT_EXPECTS(
          appr_algo->size_links_per_element_ == (extent + remainder) * sizeof(IdxT) + sizeof(int),
          "Size links per element mismatch");
      }
    }

    const auto end_clock = std::chrono::system_clock::now();
    if (i > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      float throughput      = bytes_written / GiB / time;
      float rows_throughput = i / time;
      float ETA             = (n_rows - i) / rows_throughput;
      RAFT_LOG_INFO(
        "# Writing rows %12lu / %12lu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f GiB\r",
        i,
        n_rows,
        i / static_cast<double>(n_rows) * 100,
        throughput,
        int(ETA / 60),
        std::fmod(ETA, 60.0f),
        bytes_written / GiB);
    }
  }

  os.flush();
  os_raw.flush();
  if (!os_raw.good()) { RAFT_LOG_WARN("Output stream is not in good state after serialization"); }

  auto end_time = std::chrono::system_clock::now();
  auto elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  RAFT_LOG_INFO("HNSW serialization with original dataset order complete in %ld ms", elapsed_time);
}

template <typename T, HnswHierarchy hierarchy>
std::enable_if_t<hierarchy == HnswHierarchy::GPU, std::unique_ptr<index<T>>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::from_cagra<GPU>");
  auto stream = raft::resource::get_cuda_stream(res);
  auto num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;

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

  const T* source_dataset = nullptr;
  int64_t n_rows, dim, source_stride;
  bool device_copy;
  if (dataset.has_value()) {
    n_rows         = dataset->extent(0);
    dim            = dataset->extent(1);
    device_copy    = false;
    source_dataset = dataset->data_handle();
    source_stride  = dim;
  } else if (auto cagra_dataset = cagra_index.dataset(); cagra_dataset.data_handle() != nullptr) {
    n_rows         = cagra_dataset.extent(0);
    dim            = cagra_dataset.extent(1);
    device_copy    = true;
    source_dataset = cagra_dataset.data_handle();
    source_stride  = cagra_dataset.stride(0);
  } else {
    RAFT_FAIL("hnsw::from_cagra<GPU>: No dataset provided");
  }

  // initialize HNSW index
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, cagra_index.metric(), hierarchy);
  auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(),
    n_rows,
    (cagra_index.graph().extent(1) + 1) / 2,
    params.ef_construction);
  appr_algo->cur_element_count = n_rows;

  // Initialize linked lists
  auto& levels = appr_algo->element_levels_;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> block_scope(
      "parallel::initialize_data<%s>(%d threads)", device_copy ? "device" : "host", num_threads);
    /* Note: batching

    If the dataset is on the device, we want to copy it to HNSW in parallel to the rest of the
    initialization loop. Ideally, we could use cudaMemcpy2DAsync to do this, but this call is very
    likely to sync with the CPU, because normally the allocated host memory is paged. To avoid this,
    we use double-buffering and copy data in small batches via pinned memory. Hence, the cuda
    device-to-host copy is completely overlapped with the host loop.

    The batching is completely disabled if the source dataset is on the host.
    */
    const int64_t max_batch_size =
      device_copy ? raft::div_rounding_up_safe<int64_t>(64 * 1024 * 1024, source_stride * sizeof(T))
                  : n_rows;
    T* bufs[2]                                                  = {nullptr, nullptr};
    std::optional<raft::pinned_matrix<T, int64_t>> bufs_storage = std::nullopt;
    if (device_copy) {
      bufs_storage.emplace(
        std::move(raft::make_pinned_matrix<T, int64_t>(res, max_batch_size * 2, source_stride)));
      bufs[0] = bufs_storage->data_handle();
      bufs[1] = bufs[0] + max_batch_size * source_stride;
    }
    auto n_batches = raft::div_rounding_up_safe<int64_t>(n_rows, max_batch_size);
    for (int64_t batch_i = -1; batch_i < n_batches; batch_i++) {
      if (device_copy) {
        if (batch_i >= 0) {
          // Sync previous batch load
          raft::resource::sync_stream(res);
        }
        auto next_batch_i = batch_i + 1;
        if (next_batch_i < n_batches) {
          auto offset     = next_batch_i * max_batch_size;
          auto batch_size = std::min(max_batch_size, n_rows - offset);
          raft::copy(
            res,
            raft::make_host_vector_view(bufs[next_batch_i % 2], batch_size * source_stride),
            raft::make_device_vector_view(source_dataset + offset * source_stride,
                                          batch_size * source_stride));
        }
      }
      if (batch_i < 0) { continue; }
      const auto i0 = batch_i * max_batch_size;
      const auto i1 = std::min(i0 + max_batch_size, n_rows);
#pragma omp parallel for num_threads(num_threads)
      for (int64_t i = i0; i < i1; i++) {
        // clear the storage (TODO: it's not clear if this is necessary)
        memset(appr_algo->get_linklist0(i), 0, appr_algo->size_links_level0_);
        // copy the data section
        auto* source_ptr = device_copy ? bufs[batch_i % 2] + (i - i0) * source_stride
                                       : source_dataset + i * source_stride;
        memcpy(appr_algo->getDataByInternalId(i), source_ptr, appr_algo->data_size_);
        // As we build the index from scratch, we assign labels 1-1 in data order
        *appr_algo->getExternalLabeLp(i) = static_cast<hnswlib::labeltype>(i);
        int32_t curlevel                 = appr_algo->getRandomLevel(appr_algo->mult_);
        levels[i]                        = curlevel;
        if (curlevel) {
          appr_algo->linkLists_[i] =
            (char*)malloc(appr_algo->size_links_per_element_ * curlevel + 1);
          if (appr_algo->linkLists_[i] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
          memset(appr_algo->linkLists_[i], 0, appr_algo->size_links_per_element_ * curlevel + 1);
        }
      }
    }
  }

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
                reinterpret_cast<char*>(&host_query_set(i - start_idx, 0)));
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

  auto graph_ptr = cagra_index.graph().data_handle();
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, graph_ptr));
  bool is_host_accessible = false;
  int64_t degree          = cagra_index.graph().extent(1);
  if (attr.type == cudaMemoryTypeUnregistered) {
    is_host_accessible = true;
  } else if (attr.hostPointer != nullptr) {
    graph_ptr          = static_cast<uint32_t*>(attr.hostPointer);
    is_host_accessible = true;
  }

  // copy cagra graph to hnswlib base layer
  if (is_host_accessible) {
    common::nvtx::range<common::nvtx::domain::cuvs> copy_scope("get_linklist0<host>");
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < n_rows; i++) {
      auto ll_i = appr_algo->get_linklist0(i);
      appr_algo->setListCount(ll_i, degree);
      auto* data = (uint32_t*)(ll_i + 1);
      for (int64_t j = 0; j < degree; j++) {
        data[j] = graph_ptr[i * degree + j];
      }
    }
  } else {
    common::nvtx::range<common::nvtx::domain::cuvs> copy_scope("get_linklist0<device>");
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(appr_algo->get_linklist0(0) + 1,
                                    appr_algo->size_data_per_element_,
                                    graph_ptr,
                                    degree * sizeof(uint32_t),
                                    degree * sizeof(uint32_t),
                                    n_rows,
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(res)));
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < n_rows; i++) {
      appr_algo->setListCount(appr_algo->get_linklist0(i), degree);
    }
    raft::resource::sync_stream(res);
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
  // special treatment for index on disk
  if (cagra_index.dataset_fd().has_value() && cagra_index.graph_fd().has_value()) {
    // Get directory from graph file descriptor
    const auto& graph_fd = cagra_index.graph_fd();
    RAFT_EXPECTS(graph_fd.has_value() && graph_fd->is_valid(),
                 "Graph file descriptor is not available for disk-backed index");

    std::string graph_path = graph_fd->get_path();
    RAFT_EXPECTS(!graph_path.empty(), "Unable to get path from graph file descriptor");

    std::string index_directory = std::filesystem::path(graph_path).parent_path().string();
    RAFT_EXPECTS(
      std::filesystem::exists(index_directory) && std::filesystem::is_directory(index_directory),
      "Directory '%s' does not exist",
      index_directory.c_str());
    std::string index_filename =
      (std::filesystem::path(index_directory) / "hnsw_index.bin").string();

    std::ofstream of(index_filename, std::ios::out | std::ios::binary);

    RAFT_EXPECTS(of, "Cannot open file %s", index_filename.c_str());

    if (dataset.has_value()) {
      serialize_to_hnswlib_with_original_dataset(res, of, params, cagra_index, dataset.value());
    } else {
      serialize_to_hnswlib_from_disk(res, of, params, cagra_index);
    }

    of.close();
    RAFT_EXPECTS(of, "Error writing output %s", index_filename.c_str());

    // Create an empty HNSW index that holds the file descriptor
    auto hnsw_index =
      std::make_unique<index_impl<T>>(cagra_index.dim(), cagra_index.metric(), params.hierarchy);

    // Open file descriptor for the HNSW index file and transfer ownership to the index
    hnsw_index->set_file_descriptor(cuvs::util::file_descriptor(index_filename, O_RDONLY));

    RAFT_LOG_INFO("HNSW index written to disk at: %s", index_filename.c_str());

    return hnsw_index;
  }

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
  // If the index is disk-backed, load it into memory first
  auto* idx_impl = dynamic_cast<index_impl<T>*>(&idx);
  if (idx_impl) { idx_impl->ensure_loaded(); }

  auto* hnswlib_index = reinterpret_cast<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>*>(
    const_cast<void*>(idx.get_index()));
  auto current_element_count = hnswlib_index->getCurrentElementCount();
  auto new_element_count     = additional_dataset.extent(0);
  [[maybe_unused]] auto num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;

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
  // If the index is disk-backed, load it into memory first
  auto* idx_impl = dynamic_cast<const index_impl<T>*>(&idx);
  if (idx_impl) { idx_impl->ensure_loaded(); }

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
  auto* idx_impl = dynamic_cast<const index_impl<T>*>(&idx);

  // Check if this is a disk-based index (created from disk-backed CAGRA)
  if (idx_impl && idx_impl->file_descriptor().has_value()) {
    // For disk-based indexes, copy the existing file to the new location
    std::string source_path = idx_impl->file_path();
    RAFT_EXPECTS(!source_path.empty(), "Disk-based index has invalid file path");
    RAFT_EXPECTS(std::filesystem::exists(source_path),
                 "Disk-based index file does not exist: %s",
                 source_path.c_str());

    // Copy the file to the new location
    std::filesystem::copy_file(
      source_path, filename, std::filesystem::copy_options::overwrite_existing);
    RAFT_LOG_INFO(
      "Copied disk-based HNSW index from %s to %s", source_path.c_str(), filename.c_str());
    return;
  }

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
  try {
    auto hnsw_index = std::make_unique<index_impl<T>>(dim, metric, params.hierarchy);
    auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
      hnsw_index->get_space(), filename);
    if (params.hierarchy == HnswHierarchy::NONE) { appr_algo->base_layer_only = true; }
    hnsw_index->set_index(std::move(appr_algo));
    *idx = hnsw_index.release();
  } catch (const std::bad_alloc& e) {
    RAFT_FAIL(
      "Failed to deserialize HNSW index from '%s': insufficient host memory. "
      "The index is too large to fit in available RAM. "
      "Consider using a machine with more memory or reducing the dataset size.",
      filename.c_str());
  }
}

/**
 * @brief Build an HNSW index on the GPU using CAGRA graph building algorithm
 *
 * This function builds an HNSW index
 * 1. Converting HNSW parameters to CAGRA parameters (ACE configuration by default)
 * 2. Building a CAGRA index
 * 3. Converting the CAGRA index to HNSW format
 */
template <typename T>
std::unique_ptr<index<T>> build(raft::resources const& res,
                                const index_params& params,
                                raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::build<ACE>");

  // Use provided ACE parameters or default ones if not specified
  auto ace_params =
    std::holds_alternative<graph_build_params::ace_params>(params.graph_build_params)
      ? std::get<graph_build_params::ace_params>(params.graph_build_params)
      : graph_build_params::ace_params{};

  // Create CAGRA index parameters from HNSW parameters
  cuvs::neighbors::cagra::index_params cagra_params;
  cagra_params.metric                    = params.metric;
  cagra_params.intermediate_graph_degree = params.M * 3;
  cagra_params.graph_degree              = params.M * 2;

  // Configure ACE parameters for CAGRA
  cuvs::neighbors::cagra::graph_build_params::ace_params cagra_ace_params;
  cagra_ace_params.npartitions        = ace_params.npartitions;
  cagra_ace_params.ef_construction    = params.ef_construction;
  cagra_ace_params.build_dir          = ace_params.build_dir;
  cagra_ace_params.use_disk           = ace_params.use_disk;
  cagra_ace_params.max_host_memory_gb = ace_params.max_host_memory_gb;
  cagra_ace_params.max_gpu_memory_gb  = ace_params.max_gpu_memory_gb;
  cagra_params.graph_build_params     = cagra_ace_params;

  RAFT_LOG_INFO(
    "hnsw::build - Building HNSW index using ACE with %zu partitions, ef_construction=%zu",
    ace_params.npartitions,
    ace_params.ef_construction);

  // Build CAGRA index using ACE
  auto cagra_index = cuvs::neighbors::cagra::build(res, cagra_params, dataset);

  RAFT_LOG_INFO("hnsw::build - Converting CAGRA index to HNSW format");

  // Convert CAGRA index to HNSW index. The resulting HNSW index uses the partitioned ACE index order.
  // See `cagra::build` and `hnsw::from_cagra` for more details on how to remap the graph to original ids.
  return from_cagra<T>(res, params, cagra_index, std::nullopt);
}

}  // namespace cuvs::neighbors::hnsw::detail
