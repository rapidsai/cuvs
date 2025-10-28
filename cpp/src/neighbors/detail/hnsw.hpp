/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../../core/omp_wrapper.hpp"
#include "../../util/file_io.hpp"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/pinned_mdarray.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

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

  RAFT_EXPECTS(index_.on_disk(), "Function only implements serialization from disk.");
  RAFT_EXPECTS(params.hierarchy != HnswHierarchy::CPU,
               "Disk2disk serialization not supported for CPU hierarchy.");

  auto n_rows           = index_.size();
  auto dim              = index_.dim();
  auto graph_degree_int = static_cast<int>(index_.graph_degree());
  RAFT_LOG_INFO("Saving CAGRA index to hnswlib format, size %zu, dim %zu, graph_degree %zu",
                static_cast<size_t>(n_rows),
                static_cast<size_t>(dim),
                static_cast<size_t>(graph_degree_int));

  auto index_directory = index_.file_directory();
  ASSERT(std::filesystem::exists(index_directory) && std::filesystem::is_directory(index_directory),
         "Directory '%s' does not exist",
         index_directory.c_str());

  std::string graph_filename =
    (std::filesystem::path(index_directory) / "cagra_graph.npy").string();
  ASSERT(std::filesystem::exists(graph_filename),
         "Graph file '%s' does not exist.",
         graph_filename.c_str());
  size_t graph_header_size = 0;
  size_t graph_n_rows      = 0;
  size_t graph_n_cols      = 0;
  {
    std::ifstream is(graph_filename, std::ios::in | std::ios::binary);
    ASSERT(is, "Cannot open graph file %s", graph_filename.c_str());
    auto start_pos    = is.tellg();
    auto header       = raft::detail::numpy_serializer::read_header(is);
    graph_header_size = static_cast<size_t>(is.tellg() - start_pos);
    ASSERT(
      header.shape.size() == 2, "Graph file should be 2D, got %zu dimensions", header.shape.size());

    graph_n_rows = header.shape[0];
    graph_n_cols = header.shape[1];
    RAFT_LOG_DEBUG("Graph file: %zu x %zu, dtype=%c, header size: %zu bytes",
                   graph_n_rows,
                   graph_n_cols,
                   header.dtype.kind,
                   graph_header_size);
  }

  std::string dataset_filename =
    (std::filesystem::path(index_directory) / "reordered_dataset.npy").string();
  ASSERT(std::filesystem::exists(dataset_filename),
         "Dataset file '%s' does not exist.",
         dataset_filename.c_str());
  size_t dataset_header_size = 0;
  size_t dataset_n_rows      = 0;
  size_t dataset_n_cols      = 0;
  {
    std::ifstream is(dataset_filename, std::ios::in | std::ios::binary);
    ASSERT(is, "Cannot open dataset file %s", dataset_filename.c_str());
    auto start_pos      = is.tellg();
    auto header         = raft::detail::numpy_serializer::read_header(is);
    dataset_header_size = static_cast<size_t>(is.tellg() - start_pos);
    ASSERT(header.shape.size() == 2,
           "Dataset file should be 2D, got %zu dimensions",
           header.shape.size());

    dataset_n_rows = header.shape[0];
    dataset_n_cols = header.shape[1];
    RAFT_LOG_DEBUG("Dataset file: %zu x %zu, dtype=%c, header size: %zu bytes",
                   dataset_n_rows,
                   dataset_n_cols,
                   header.dtype.kind,
                   dataset_header_size);
  }

  std::string label_filename =
    (std::filesystem::path(index_directory) / "dataset_mapping.npy").string();
  ASSERT(std::filesystem::exists(label_filename),
         "Label file '%s' does not exist.",
         label_filename.c_str());
  size_t label_header_size = 0;
  size_t label_n_elements  = 0;
  {
    std::ifstream is(label_filename, std::ios::in | std::ios::binary);
    ASSERT(is, "Cannot open label file %s", label_filename.c_str());
    auto start_pos    = is.tellg();
    auto header       = raft::detail::numpy_serializer::read_header(is);
    label_header_size = static_cast<size_t>(is.tellg() - start_pos);
    ASSERT(
      header.shape.size() == 1, "Label file should be 1D, got %zu dimensions", header.shape.size());

    label_n_elements = header.shape[0];
    RAFT_LOG_DEBUG("Label file: %zu elements, dtype=%c, header size: %zu bytes",
                   label_n_elements,
                   header.dtype.kind,
                   label_header_size);
  }

  // Verify consistency
  ASSERT(graph_n_rows == static_cast<size_t>(n_rows),
         "Graph rows (%zu) != index size (%zu)",
         graph_n_rows,
         static_cast<size_t>(n_rows));
  ASSERT(dataset_n_rows == static_cast<size_t>(n_rows),
         "Dataset rows (%zu) != index size (%zu)",
         dataset_n_rows,
         static_cast<size_t>(n_rows));
  ASSERT(label_n_elements == static_cast<size_t>(n_rows),
         "Label elements (%zu) != index size (%zu)",
         label_n_elements,
         static_cast<size_t>(n_rows));
  ASSERT(graph_n_cols == static_cast<size_t>(graph_degree_int),
         "Graph cols (%zu) != graph degree (%d)",
         graph_n_cols,
         graph_degree_int);
  ASSERT(dataset_n_cols == static_cast<size_t>(dim),
         "Dataset cols (%zu) != dimensions (%zu)",
         dataset_n_cols,
         static_cast<size_t>(dim));

  // Open file descriptors for batched reading
  int graph_fd = open(graph_filename.c_str(), O_RDONLY);
  ASSERT(graph_fd >= 0, "Failed to open graph file %s", graph_filename.c_str());

  int dataset_fd = open(dataset_filename.c_str(), O_RDONLY);
  ASSERT(dataset_fd >= 0, "Failed to open dataset file %s", dataset_filename.c_str());

  int label_fd = open(label_filename.c_str(), O_RDONLY);
  ASSERT(label_fd >= 0, "Failed to open label file %s", label_filename.c_str());

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

  assert(appr_algo->size_data_per_element_ ==
         dim * sizeof(T) + appr_algo->maxM0_ * sizeof(IdxT) + sizeof(int) + sizeof(size_t));

  RAFT_LOG_INFO("Writing base level");
  size_t bytes_written = 0;
  float GiB            = 1 << 30;
  IdxT zero            = 0;
  assert(appr_algo->size_data_per_element_ ==
         dim * sizeof(T) + appr_algo->maxM0_ * sizeof(IdxT) + sizeof(int) + sizeof(size_t));

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
        ASSERT(bytes_read == static_cast<ssize_t>(graph_bytes),
               "Failed to read graph data: expected %zu, got %zd",
               graph_bytes,
               bytes_read);
      }
#pragma omp section
      {
        ssize_t bytes_read =
          pread(dataset_fd, dataset_buffer.data_handle(), dataset_bytes, dataset_offset);
        ASSERT(bytes_read == static_cast<ssize_t>(dataset_bytes),
               "Failed to read dataset data: expected %zu, got %zd",
               dataset_bytes,
               bytes_read);
      }
#pragma omp section
      {
        ssize_t bytes_read = pread(label_fd, label_buffer.data_handle(), label_bytes, label_offset);
        ASSERT(bytes_read == static_cast<ssize_t>(label_bytes),
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
        assert(odd_graph_degree == appr_algo->maxM0_ - graph_degree_int);
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

  // Close file descriptors
  close(graph_fd);
  close(dataset_fd);
  close(label_fd);

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
        assert(appr_algo->size_links_per_element_ ==
               (neighbor_view.extent(1) + remainder) * sizeof(IdxT) + sizeof(int));
      }
    }

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
          raft::copy(bufs[next_batch_i % 2],
                     source_dataset + offset * source_stride,
                     batch_size * source_stride,
                     stream);
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
  if (cagra_index.on_disk()) {
    auto index_directory = cagra_index.file_directory();
    ASSERT(
      std::filesystem::exists(index_directory) && std::filesystem::is_directory(index_directory),
      "Directory '%s' does not exist",
      index_directory.c_str());
    std::string index_filename =
      (std::filesystem::path(index_directory) / "hnsw_index.bin").string();

    std::ofstream of(index_filename, std::ios::out | std::ios::binary);

    if (!of) { RAFT_FAIL("Cannot open file %s", index_filename.c_str()); }

    serialize_to_hnswlib_from_disk(res, of, params, cagra_index);

    of.close();
    if (!of) { RAFT_FAIL("Error writing output %s", index_filename.c_str()); }

    return nullptr;
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
