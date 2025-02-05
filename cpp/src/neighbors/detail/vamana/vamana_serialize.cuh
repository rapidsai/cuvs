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

#include "vamana_structs.cuh"

#include <cuvs/neighbors/vamana.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>

#include "../dataset_serialize.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <type_traits>

namespace cuvs::neighbors::vamana::detail {

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res the raft resource handle
 * @param[in] file_name the path and name of the DiskAN index file generated
 * @param[in] index_ VAMANA index
 * @param[in] include_dataset whether to include the dataset in the serialized output
 *
 */

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& file_name,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  // Write graph to first index file (format from MSFT DiskANN OSS)
  std::ofstream index_of(file_name, std::ios::out | std::ios::binary);
  if (!index_of) { RAFT_FAIL("Cannot open file %s", file_name.c_str()); }

  size_t file_offset = 0;
  index_of.seekp(file_offset, index_of.beg);
  uint32_t max_degree          = 0;
  size_t index_size            = 24;  // Starting metadata
  uint32_t start               = static_cast<uint32_t>(index_.medoid());
  size_t num_frozen_points     = 0;
  uint32_t max_observed_degree = 0;

  index_of.write((char*)&index_size, sizeof(uint64_t));
  index_of.write((char*)&max_observed_degree, sizeof(uint32_t));
  index_of.write((char*)&start, sizeof(uint32_t));
  index_of.write((char*)&num_frozen_points, sizeof(size_t));

  auto d_graph = index_.graph();
  auto h_graph = raft::make_host_matrix<IdxT, int64_t>(d_graph.extent(0), d_graph.extent(1));
  raft::copy(h_graph.data_handle(),
             d_graph.data_handle(),
             d_graph.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  size_t total_edges = 0;
  size_t num_sparse  = 0;
  size_t num_single  = 0;

  for (uint32_t i = 0; i < h_graph.extent(0); i++) {
    uint32_t node_edges = 0;
    for (; node_edges < h_graph.extent(1); node_edges++) {
      if (h_graph(i, node_edges) == raft::upper_bound<IdxT>()) { break; }
    }

    if (node_edges < 3) num_sparse++;
    if (node_edges < 2) num_single++;
    total_edges += node_edges;

    index_of.write((char*)&node_edges, sizeof(uint32_t));
    if constexpr (!std::is_same_v<IdxT, uint32_t>) {
      RAFT_FAIL("serialization is only implemented for uint32_t graph");
    }
    index_of.write((char*)&h_graph(i, 0), node_edges * sizeof(uint32_t));

    max_degree = node_edges > max_degree ? (uint32_t)node_edges : max_degree;
    index_size += (size_t)(sizeof(uint32_t) * (node_edges + 1));
  }
  index_of.seekp(file_offset, index_of.beg);
  index_of.write((char*)&index_size, sizeof(uint64_t));
  index_of.write((char*)&max_degree, sizeof(uint32_t));

  RAFT_LOG_DEBUG(
    "Wrote file out, index size:%lu, max_degree:%u, num_sparse:%ld, num_single:%ld, total "
    "edges:%ld, avg degree:%f",
    index_size,
    max_degree,
    num_sparse,
    num_single,
    total_edges,
    (float)total_edges / (float)h_graph.extent(0));

  index_of.close();
  if (!index_of) { RAFT_FAIL("Error writing output %s", file_name.c_str()); }

  if (include_dataset) {
    // try allocating a buffer for the dataset on host
    try {
      const cuvs::neighbors::strided_dataset<T, int64_t>* strided_dataset =
        dynamic_cast<cuvs::neighbors::strided_dataset<T, int64_t>*>(
          const_cast<cuvs::neighbors::dataset<int64_t>*>(&index_.data()));
      if (strided_dataset == nullptr) {
        RAFT_LOG_DEBUG("dynamic_cast to strided_dataset failed");
      } else {
        auto h_dataset =
          raft::make_host_matrix<T, int64_t>(strided_dataset->n_rows(), strided_dataset->dim());
        raft::copy(h_dataset.data_handle(),
                   strided_dataset->view().data_handle(),
                   strided_dataset->n_rows() * strided_dataset->dim(),
                   raft::resource::get_cuda_stream(res));
        std::string dataset_base_file = file_name + ".data";
        std::ofstream dataset_of(dataset_base_file, std::ios::out | std::ios::binary);
        if (!dataset_of) { RAFT_FAIL("Cannot open file %s", dataset_base_file.c_str()); }
        size_t dataset_file_offset = 0;
        int size                   = static_cast<int>(index_.size());
        int dim                    = static_cast<int>(index_.dim());
        dataset_of.seekp(dataset_file_offset, dataset_of.beg);
        dataset_of.write((char*)&size, sizeof(int));
        dataset_of.write((char*)&dim, sizeof(int));
        for (int i = 0; i < size; i++) {
          dataset_of.write((char*)(h_dataset.data_handle() + i * h_dataset.extent(1)),
                           dim * sizeof(T));
        }
        dataset_of.close();
        if (!dataset_of) { RAFT_FAIL("Error writing output %s", dataset_base_file.c_str()); }
      }
    } catch (std::bad_alloc& e) {
      RAFT_LOG_INFO("Failed to serialize dataset");
    } catch (raft::logic_error& e) {
      RAFT_LOG_INFO("Failed to serialize dataset");
    }
  }
}

}  // namespace cuvs::neighbors::vamana::detail
