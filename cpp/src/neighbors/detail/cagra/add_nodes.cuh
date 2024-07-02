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
#include "../ann_utils.cuh"
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/stats/histogram.cuh>

#include <rmm/device_buffer.hpp>

#include <omp.h>

#include <cstdint>

static const std::string RAFT_NAME = "raft";

namespace cuvs::neighbors::cagra {
template <class T, class IdxT, class Accessor>
void add_node_core(
  raft::resources const& handle,
  const cuvs::neighbors::cagra::index<T, IdxT>& idx,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::layout_stride, Accessor>
    additional_dataset_view,
  raft::host_matrix_view<IdxT, std::int64_t> updated_graph)
{
  using DistanceT                 = float;
  const std::size_t degree        = idx.graph_degree();
  const std::size_t dim           = idx.dim();
  const std::size_t old_size      = idx.dataset().extent(0);
  const std::size_t num_add       = additional_dataset_view.extent(0);
  const std::size_t new_size      = old_size + num_add;
  const std::uint32_t base_degree = degree * 2;

  // Step 0: Calculate the number of incoming edges for each node
  auto dev_num_incoming_edges = raft::make_device_vector<int, std::uint64_t>(handle, new_size);

  RAFT_CUDA_TRY(cudaMemsetAsync(dev_num_incoming_edges.data_handle(),
                                0,
                                sizeof(int) * new_size,
                                raft::resource::get_cuda_stream(handle)));
  raft::stats::histogram<IdxT, std::int64_t>(raft::stats::HistTypeAuto,
                                             dev_num_incoming_edges.data_handle(),
                                             old_size,
                                             idx.graph().data_handle(),
                                             old_size * degree,
                                             1,
                                             raft::resource::get_cuda_stream(handle));

  auto host_num_incoming_edges = raft::make_host_vector<int, std::uint64_t>(new_size);
  raft::copy(host_num_incoming_edges.data_handle(),
             dev_num_incoming_edges.data_handle(),
             new_size,
             raft::resource::get_cuda_stream(handle));

  const std::size_t max_chunk_size = 1024;

  cuvs::neighbors::cagra::search_params params;
  params.itopk_size = std::max(base_degree * 2lu, 256lu);

  // Memory space for rank-based neighbor list
  auto mr = raft::resource::get_workspace_resource(handle);

  auto neighbor_indices = raft::make_device_mdarray<IdxT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, base_degree));

  auto neighbor_distances = raft::make_device_mdarray<DistanceT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, base_degree));

  auto queries = raft::make_device_mdarray<T, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_chunk_size, dim));

  auto host_neighbor_indices =
    raft::make_host_matrix<IdxT, std::int64_t>(max_chunk_size, base_degree);

  cuvs::spatial::knn::detail::utils::batch_load_iterator<T> additional_dataset_batch(
    additional_dataset_view.data_handle(),
    num_add,
    additional_dataset_view.stride(0),
    max_chunk_size,
    raft::resource::get_cuda_stream(handle),
    raft::resource::get_workspace_resource(handle));
  for (const auto& batch : additional_dataset_batch) {
    // Step 1: Obtain K (=base_degree) nearest neighbors of the new vectors by CAGRA search
    // Create queries
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(queries.data_handle(),
                                    sizeof(T) * dim,
                                    batch.data(),
                                    sizeof(T) * additional_dataset_view.stride(0),
                                    sizeof(T) * dim,
                                    batch.size(),
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(handle)));

    const auto queries_view = raft::make_device_matrix_view<const T, std::int64_t>(
      queries.data_handle(), batch.size(), dim);

    auto neighbor_indices_view = raft::make_device_matrix_view<IdxT, std::int64_t>(
      neighbor_indices.data_handle(), batch.size(), base_degree);
    auto neighbor_distances_view = raft::make_device_matrix_view<float, std::int64_t>(
      neighbor_distances.data_handle(), batch.size(), base_degree);

    neighbors::cagra::search(
      handle, params, idx, queries_view, neighbor_indices_view, neighbor_distances_view);

    raft::copy(host_neighbor_indices.data_handle(),
               neighbor_indices.data_handle(),
               batch.size() * base_degree,
               raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);

    // Step 2: rank-based reordering
#pragma omp parallel
    {
      std::vector<std::pair<IdxT, std::size_t>> detourable_node_count_list(base_degree);
      for (std::size_t vec_i = omp_get_thread_num(); vec_i < batch.size();
           vec_i += omp_get_num_threads()) {
        // Count detourable edges
        for (std::uint32_t i = 0; i < base_degree; i++) {
          std::uint32_t detourable_node_count = 0;
          const auto a_id                     = host_neighbor_indices(vec_i, i);
          for (std::uint32_t j = 0; j < i; j++) {
            const auto b0_id = host_neighbor_indices(vec_i, j);
            assert(b0_id < idx.size());
            for (std::uint32_t k = 0; k < degree; k++) {
              const auto b1_id = updated_graph(b0_id, k);
              if (a_id == b1_id) {
                detourable_node_count++;
                break;
              }
            }
          }
          detourable_node_count_list[i] = std::make_pair(a_id, detourable_node_count);
        }
        std::sort(detourable_node_count_list.begin(),
                  detourable_node_count_list.end(),
                  [&](const std::pair<IdxT, std::size_t> a, const std::pair<IdxT, std::size_t> b) {
                    return a.second < b.second;
                  });

        for (std::size_t i = 0; i < degree; i++) {
          updated_graph(old_size + batch.offset() + vec_i, i) = detourable_node_count_list[i].first;
        }
      }
    }

    // Step 3: Add reverse edges
    const std::uint32_t rev_edge_search_range = degree / 2;
    const std::uint32_t num_rev_edges         = degree / 2;
    std::vector<IdxT> rev_edges(num_rev_edges), temp(degree);
    for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
      // Create a reverse edge list
      const auto target_new_node_id = old_size + batch.offset() + vec_i;
      for (std::size_t i = 0; i < num_rev_edges; i++) {
        const auto target_node_id = updated_graph(old_size + batch.offset() + vec_i, i);

        IdxT replace_id                        = new_size;
        IdxT replace_id_j                      = 0;
        std::size_t replace_num_incoming_edges = 0;
        for (std::int32_t j = degree - 1; j >= static_cast<std::int32_t>(rev_edge_search_range);
             j--) {
          const auto neighbor_id               = updated_graph(target_node_id, j);
          const std::size_t num_incoming_edges = host_num_incoming_edges(neighbor_id);
          if (num_incoming_edges > replace_num_incoming_edges) {
            // Check duplication
            bool dup = false;
            for (std::uint32_t k = 0; k < i; k++) {
              if (rev_edges[k] == neighbor_id) {
                dup = true;
                break;
              }
            }
            if (dup) { continue; }

            // Update rev edge candidate
            replace_num_incoming_edges = num_incoming_edges;
            replace_id                 = neighbor_id;
            replace_id_j               = j;
          }
        }
        if (replace_id >= new_size) {
          std::fprintf(stderr, "Invalid rev edge index (%u)\n", replace_id);
          return;
        }
        updated_graph(target_node_id, replace_id_j) = target_new_node_id;
        rev_edges[i]                                = replace_id;
      }
      host_num_incoming_edges(target_new_node_id) = num_rev_edges;

      // Create a neighbor list of a new node by interleaving the kNN neighbor list and reverse edge
      // list
      std::uint32_t interleave_switch = 0, rank_base_i = 0, rev_edges_return_i = 0, num_add = 0;
      const auto rank_based_list_ptr =
        updated_graph.data_handle() + (old_size + batch.offset() + vec_i) * degree;
      const auto rev_edges_return_list_ptr = rev_edges.data();
      while (num_add < degree) {
        const auto node_list_ptr =
          interleave_switch == 0 ? rank_based_list_ptr : rev_edges_return_list_ptr;
        auto& node_list_index          = interleave_switch == 0 ? rank_base_i : rev_edges_return_i;
        const auto max_node_list_index = interleave_switch == 0 ? degree : num_rev_edges;
        for (; node_list_index < max_node_list_index; node_list_index++) {
          const auto candidate = node_list_ptr[node_list_index];
          // Check duplication
          bool dup = false;
          for (std::uint32_t j = 0; j < num_add; j++) {
            if (temp[j] == candidate) {
              dup = true;
              break;
            }
          }
          if (!dup) {
            temp[num_add] = candidate;
            num_add++;
            break;
          }
        }
        interleave_switch = 1 - interleave_switch;
      }
      for (std::uint32_t i = 0; i < degree; i++) {
        updated_graph(target_new_node_id, i) = temp[i];
      }
    }
  }
}

template <class T, class IdxT>
void add_graph_nodes(
  raft::resources const& handle,
  raft::device_matrix_view<const T, int64_t, raft::layout_stride> input_updated_dataset_view,
  const neighbors::cagra::index<T, IdxT>& index,
  raft::host_matrix_view<IdxT, std::int64_t> updated_graph_view,
  const cagra::extend_params& params)
{
  assert(input_updated_dataset_view.extent(0) >= index.size());

  const std::size_t initial_dataset_size = index.size();
  const std::size_t new_dataset_size     = input_updated_dataset_view.extent(0);
  const std::size_t num_new_nodes        = new_dataset_size - initial_dataset_size;
  const std::size_t degree               = index.graph_degree();
  const std::size_t dim                  = index.dim();
  const std::size_t stride               = input_updated_dataset_view.stride(0);
  const std::size_t max_chunk_size_      = params.max_chunk_size == 0 ? 1 : params.max_chunk_size;

  raft::copy(updated_graph_view.data_handle(),
             index.graph().data_handle(),
             index.graph().size(),
             raft::resource::get_cuda_stream(handle));

  neighbors::cagra::index<T, IdxT> internal_index(
    handle,
    index.metric(),
    raft::make_device_matrix_view<const T, int64_t>(nullptr, 0, dim),
    raft::make_device_matrix_view<const IdxT, int64_t>(nullptr, 0, degree));

  for (std::size_t additional_dataset_offset = 0; additional_dataset_offset < num_new_nodes;
       additional_dataset_offset += max_chunk_size_) {
    const auto actual_chunk_size =
      std::min(num_new_nodes - additional_dataset_offset, max_chunk_size_);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, std::int64_t>(
      input_updated_dataset_view.data_handle(),
      initial_dataset_size + additional_dataset_offset,
      dim,
      stride);
    auto graph_view = raft::make_host_matrix_view<const IdxT, std::int64_t>(
      updated_graph_view.data_handle(), initial_dataset_size + additional_dataset_offset, degree);

    internal_index.update_dataset(handle, dataset_view);

    // Note: The graph is copied to the device memory.
    internal_index.update_graph(handle, graph_view);
    raft::resource::sync_stream(handle);

    auto updated_graph = raft::make_host_matrix_view<IdxT, std::int64_t>(
      updated_graph_view.data_handle(),
      initial_dataset_size + additional_dataset_offset + actual_chunk_size,
      degree);
    auto additional_dataset_view = raft::make_device_strided_matrix_view<const T, std::int64_t>(
      input_updated_dataset_view.data_handle() +
        (initial_dataset_size + additional_dataset_offset) * stride,
      actual_chunk_size,
      dim,
      stride);

    neighbors::cagra::add_node_core<T, IdxT>(
      handle, internal_index, additional_dataset_view, updated_graph);
    raft::resource::sync_stream(handle);
  }
}

template <class T, class IdxT, class Accessor>
void extend_core(
  raft::resources const& handle,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> additional_dataset,
  cuvs::neighbors::cagra::index<T, IdxT>& index,
  const cagra::extend_params& params,
  std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> new_dataset_buffer_view,
  std::optional<raft::device_matrix_view<IdxT, int64_t>> new_graph_buffer_view)
{
  if (dynamic_cast<const non_owning_dataset<T, IdxT>*>(&index.data()) != nullptr &&
      !new_dataset_buffer_view.has_value()) {
    RAFT_LOG_WARN(
      "New memory space for extended dataset will be allocated while the memory space for the old "
      "dataset is allocated by user.");
  }
  const std::size_t num_new_nodes        = additional_dataset.extent(0);
  const std::size_t initial_dataset_size = index.size();
  const std::size_t new_dataset_size     = initial_dataset_size + num_new_nodes;
  const std::size_t degree               = index.graph_degree();
  const std::size_t dim                  = index.dim();

  if (new_dataset_buffer_view.has_value() &&
      static_cast<std::size_t>(new_dataset_buffer_view.value().extent(0)) != new_dataset_size) {
    RAFT_LOG_ERROR(
      "The extended dataset size (%lu) must be the initial dataset size (%lu) + additional dataset "
      "size (%lu). "
      "Please fix the memory buffer size for the extended dataset.",
      new_dataset_buffer_view.value().extent(0),
      initial_dataset_size,
      num_new_nodes);
  }

  if (new_graph_buffer_view.has_value() &&
      static_cast<std::size_t>(new_graph_buffer_view.value().extent(0)) != new_dataset_size) {
    RAFT_LOG_ERROR(
      "The extended graph size (%lu) must be the initial dataset size (%lu) + additional dataset "
      "size (%lu). "
      "Please fix the memory buffer size for the extended graph.",
      new_graph_buffer_view.value().extent(0),
      initial_dataset_size,
      num_new_nodes);
  }

  using ds_idx_type = decltype(index.data().n_rows());
  if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
      strided_dset != nullptr) {
    // Allocate memory space for updated graph on host
    auto updated_graph = raft::make_host_matrix<IdxT, std::int64_t>(new_dataset_size, degree);

    const auto stride    = strided_dset->stride();
    auto updated_dataset = raft::make_device_matrix<T, std::int64_t>(handle, 0, stride);
    auto updated_dataset_view =
      raft::make_device_strided_matrix_view<T, std::int64_t>(nullptr, 0, dim, stride);

    // Update dataset
    auto host_updated_dataset = raft::make_host_matrix<T, std::int64_t>(new_dataset_size, stride);

    // The padding area must be filled with zeros.!!!!!!!!!!!!!!!!!!!
    memset(host_updated_dataset.data_handle(), 0, sizeof(T) * host_updated_dataset.size());

    RAFT_CUDA_TRY(cudaMemcpy2DAsync(host_updated_dataset.data_handle(),
                                    sizeof(T) * stride,
                                    strided_dset->view().data_handle(),
                                    sizeof(T) * stride,
                                    sizeof(T) * dim,
                                    initial_dataset_size,
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(handle)));
    RAFT_CUDA_TRY(
      cudaMemcpy2DAsync(host_updated_dataset.data_handle() + initial_dataset_size * stride,
                        sizeof(T) * stride,
                        additional_dataset.data_handle(),
                        sizeof(T) * additional_dataset.stride(0),
                        sizeof(T) * dim,
                        num_new_nodes,
                        cudaMemcpyDefault,
                        raft::resource::get_cuda_stream(handle)));

    if (new_dataset_buffer_view.has_value()) {
      updated_dataset_view = new_dataset_buffer_view.value();
    } else {
      // Deallocate the current dataset memory space if the dataset is `owning'.
      index.update_dataset(
        handle, raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, dim, stride));

      // Allocate the new dataset
      updated_dataset = raft::make_device_matrix<T, std::int64_t>(handle, new_dataset_size, stride);
      updated_dataset_view = raft::make_device_strided_matrix_view<T, std::int64_t>(
        updated_dataset.data_handle(), new_dataset_size, dim, stride);
    }

    // Copy updated dataset on host memory to device memory
    raft::copy(updated_dataset_view.data_handle(),
               host_updated_dataset.data_handle(),
               new_dataset_size * stride,
               raft::resource::get_cuda_stream(handle));

    // Add graph nodes
    cuvs::neighbors::cagra::add_graph_nodes<T, IdxT>(
      handle, raft::make_const_mdspan(updated_dataset_view), index, updated_graph.view(), params);

    // Update index dataset
    if (new_dataset_buffer_view.has_value()) {
      index.update_dataset(handle, raft::make_const_mdspan(updated_dataset_view));
    } else {
      using out_mdarray_type          = decltype(updated_dataset);
      using out_layout_type           = typename out_mdarray_type::layout_type;
      using out_container_policy_type = typename out_mdarray_type::container_policy_type;
      using out_owning_type =
        owning_dataset<T, int64_t, out_layout_type, out_container_policy_type>;
      auto out_layout = raft::make_strided_layout(updated_dataset_view.extents(),
                                                  std::array<int64_t, 2>{stride, 1});

      index.update_dataset(handle, out_owning_type{std::move(updated_dataset), out_layout});
    }

    // Update index graph
    if (new_graph_buffer_view.has_value()) {
      auto device_graph_view = new_graph_buffer_view.value();
      raft::copy(device_graph_view.data_handle(),
                 updated_graph.data_handle(),
                 updated_graph.size(),
                 raft::resource::get_cuda_stream(handle));
      index.update_graph(handle, raft::make_const_mdspan(device_graph_view));
    } else {
      index.update_graph(handle, raft::make_const_mdspan(updated_graph.view()));
    }
  } else {
    RAFT_FAIL("Only uncompressed dataset is supported");
  }
}
}  // namespace cuvs::neighbors::cagra
