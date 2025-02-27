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

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) + (Y)-1) / (Y))
#define ROUND_UP(X, Y)     (((uint64_t)(X) + (Y)-1) / (Y) * (Y))

namespace {

/**
 * Parse quantizer file.
 *
 * Experimental, both the API and the file format are subject to change.
 *
 * @param[in] path the path of the DiskANN quantizer file
 *
 * @return the PQ encoding table as std::vector<float> and the OPQ matrix as std::vector<float>
 */

std::pair<std::vector<float>, std::vector<float>> parse_opq_file(const std::string& path,
                                                                 const uint32_t vector_dim,
                                                                 const uint32_t pq_dim,
                                                                 const uint32_t pq_codebook_size)
{
  std::ifstream opq_if(path, std::ios::in | std::ios::binary);
  if (!opq_if) { RAFT_FAIL("Cannot open file %s", path.c_str()); }

  // check file size
  opq_if.ignore(std::numeric_limits<std::streamsize>::max());
  uint32_t length = opq_if.gcount();
  uint32_t length_expected =
    14 + (4 * vector_dim * pq_codebook_size) + (4 * vector_dim * vector_dim);
  if (length != length_expected) {
    RAFT_FAIL("OPQ file doesn't have expected size. Expected: %s Actual: %s",
              std::to_string(length_expected).c_str(),
              std::to_string(length).c_str());
  }
  opq_if.clear();  // Since ignore will have set eof.
  opq_if.seekg(0, std::ios_base::beg);

  // check metadata
  unsigned char quantizerType, reconstructType;
  int32_t numCodebooks, entriesPerCodebook, codebookDim;

  opq_if.read((char*)&quantizerType, 1);
  if (quantizerType != 2) {
    RAFT_FAIL("Expected quantizer type 2; parsed: %s", std::to_string(quantizerType).c_str());
  }

  opq_if.read((char*)&reconstructType, 1);
  if (reconstructType != 0) {
    RAFT_FAIL("Expected reconstruct type 0; parsed: %s", std::to_string(reconstructType).c_str());
  }

  opq_if.read((char*)&numCodebooks, 4);
  if ((uint32_t)numCodebooks != pq_dim) {
    RAFT_FAIL("PQDim mismatch. Expected: %s Actual: %s",
              std::to_string(pq_dim).c_str(),
              std::to_string(numCodebooks).c_str());
  }

  opq_if.read((char*)&entriesPerCodebook, 4);
  if ((uint32_t)entriesPerCodebook != pq_codebook_size) {
    RAFT_FAIL("PQCodebookSize mismatch. Expected: %s Actual: %s",
              std::to_string(pq_codebook_size).c_str(),
              std::to_string(entriesPerCodebook).c_str());
  }

  opq_if.read((char*)&codebookDim, 4);
  if ((uint32_t)codebookDim != vector_dim / pq_dim) {
    RAFT_FAIL("Codebook dim mismatch. Expected: %s Actual: %s",
              std::to_string(vector_dim / pq_dim).c_str(),
              std::to_string(codebookDim).c_str());
  }

  // parse PQEncodingTable
  std::vector<float> PQEncodingTable;
  PQEncodingTable.reserve(vector_dim * pq_codebook_size);
  for (int i = 0; (uint32_t)i < vector_dim * pq_codebook_size; i++) {
    opq_if.read((char*)&PQEncodingTable[i],
                4);  // Reconstruct type of the overall quantizer is int8 but because it's OPQ, need
                     // to read it as float
  }

  // parse OPQMatrix
  std::vector<float> OPQMatrix;
  OPQMatrix.reserve(vector_dim * vector_dim);
  // Need to transpose the matrix during ingestion to get the right semantics
  for (int iRow = 0; (uint32_t)iRow < vector_dim; iRow++) {
    for (int iCol = 0; (uint32_t)iCol < vector_dim; iCol++) {
      opq_if.read((char*)&OPQMatrix[iCol * vector_dim + iRow], 4);
    }
  }

  return std::make_pair(PQEncodingTable, OPQMatrix);
}

}  // namespace

namespace cuvs::neighbors::vamana::detail {

/**
 * Save the index to file with sector alignment.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res the raft resource handle
 * @param[in] h_graph the host matrix representation of the VAMANA index
 * @param[in] dataset the raw dataset from which the VAMANA index is built
 * @param[in] medoid the medoid
 * @param[out] output_writer the stream object for writing the output file
 *
 */
template <typename T, typename IdxT, typename HostMatT>
void serialize_sector_aligned(raft::resources const& res,
                              const HostMatT& h_graph,
                              const cuvs::neighbors::dataset<int64_t>& dataset,
                              const uint64_t medoid,
                              std::ofstream& output_writer)
{
  if constexpr (!std::is_same_v<IdxT, uint32_t>) {
    RAFT_FAIL("serialization is only implemented for uint32_t graph");
  }

  const uint64_t sector_len = 4096;

  const uint64_t npts(dataset.n_rows()), ndims(dataset.dim());

  // stats for metadata and logging
  uint32_t max_degree = 0;
  size_t index_size   = 24;  // Starting metadata in Vamana file
  size_t total_edges  = 0;
  size_t num_sparse   = 0;
  size_t num_single   = 0;
  for (uint32_t i = 0; i < h_graph.extent(0); i++) {
    uint32_t node_edges = 0;
    for (; node_edges < h_graph.extent(1); node_edges++) {
      if (h_graph(i, node_edges) == raft::upper_bound<IdxT>()) { break; }
    }

    if (node_edges < 3) num_sparse++;
    if (node_edges < 2) num_single++;
    total_edges += node_edges;

    max_degree = max(max_degree, node_edges);
    index_size += sizeof(uint32_t) * (node_edges + 1);
  }

  const uint64_t max_node_len =
    ((static_cast<uint64_t>(max_degree) + 1) * sizeof(IdxT)) + (ndims * sizeof(T));
  const uint64_t nnodes_per_sector = sector_len / max_node_len;  // 0 if max_node_len > sector_len

  // copy dataset to host
  auto dataset_strided =
    dynamic_cast<const cuvs::neighbors::strided_dataset<T, int64_t>*>(&dataset);
  if (!dataset_strided) { RAFT_FAIL("Invalid dataset"); }
  auto d_data = dataset_strided->view();
  auto h_data = raft::make_host_matrix<T, IdxT>(npts, ndims);
  auto stride = dataset_strided->stride();
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(h_data.data_handle(),
                                  sizeof(T) * ndims,
                                  d_data.data_handle(),
                                  sizeof(T) * stride,
                                  sizeof(T) * ndims,
                                  npts,
                                  cudaMemcpyDefault,
                                  raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);

  // buffers
  std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(sector_len);
  std::unique_ptr<char[]> multisector_buf =
    std::make_unique<char[]>(ROUND_UP(max_node_len, sector_len));
  std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
  IdxT& nnbrs                      = *(IdxT*)(node_buf.get() + ndims * sizeof(T));
  IdxT* const nhood_buf            = (IdxT*)(node_buf.get() + (ndims * sizeof(T)) + sizeof(IdxT));

  const uint64_t n_sectors = nnodes_per_sector > 0 ? DIV_ROUND_UP(npts, nnodes_per_sector)
                                                   : npts * DIV_ROUND_UP(max_node_len, sector_len);
  // metadata
  const uint64_t disk_index_file_size = (n_sectors + 1) * sector_len;
  const uint64_t vamana_frozen_num{}, vamana_frozen_loc{};
  const bool append_reorder_data = false;
  std::vector<uint64_t> output_meta;
  output_meta.push_back(npts);
  output_meta.push_back(ndims);
  output_meta.push_back(medoid);
  output_meta.push_back(max_node_len);
  output_meta.push_back(nnodes_per_sector);
  output_meta.push_back(vamana_frozen_num);
  output_meta.push_back(vamana_frozen_loc);
  output_meta.push_back(static_cast<uint64_t>(append_reorder_data));
  output_meta.push_back(disk_index_file_size);

  // zero out first sector of output file
  output_writer.seekp(0, output_writer.beg);
  output_writer.write(sector_buf.get(), sector_len);
  // write metadata to first sector
  output_writer.seekp(0, output_writer.beg);
  const int metadata_size  = static_cast<int>(output_meta.size());
  const int metadata_ndims = 1;
  output_writer.write((char*)&metadata_size, sizeof(int));
  output_writer.write((char*)&metadata_ndims, sizeof(int));
  output_writer.write((char*)output_meta.data(), sizeof(uint64_t) * output_meta.size());
  output_writer.seekp(sector_len, output_writer.beg);

  if (nnodes_per_sector > 0) {
    uint64_t cur_node_id = 0;
    // Write multiple nodes per sector
    for (uint64_t sector = 0; sector < n_sectors; sector++) {
      if (sector && sector % 100000 == 0)
        std::cout << "Sector #" << sector << " written" << std::endl;
      memset(sector_buf.get(), 0, sector_len);
      for (uint64_t sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts;
           sector_node_id++) {
        memset(node_buf.get(), 0, max_node_len);
        // copy node coords to buffer
        memcpy(node_buf.get(), &h_data(cur_node_id, uint64_t(0)), ndims * sizeof(T));

        IdxT node_edges = 0;
        for (; node_edges < h_graph.extent(1); node_edges++) {
          if (h_graph(cur_node_id, node_edges) == raft::upper_bound<IdxT>()) { break; }
        }
        // write nnbrs to buffer
        nnbrs = node_edges;

        // sanity checks on nnbrs
        assert(nnbrs > 0);
        assert(nnbrs <= max_degree);

        // copy neighbors to buffer
        memcpy(nhood_buf, &h_graph(cur_node_id, 0), nnbrs * sizeof(IdxT));

        // get offset into sector_buf
        char* sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

        // copy node buf into sector_node_buf
        memcpy(sector_node_buf, node_buf.get(), max_node_len);
        cur_node_id++;
      }
      // write sector to disk
      output_writer.write(sector_buf.get(), sector_len);
    }
  } else {
    // Write multi-sector nodes
    uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, sector_len);
    for (uint64_t cur_node_id = 0; cur_node_id < npts; cur_node_id++) {
      if (cur_node_id && (cur_node_id * nsectors_per_node) % 100000 == 0)
        std::cout << "Sector #" << cur_node_id * nsectors_per_node << " written" << std::endl;
      memset(multisector_buf.get(), 0, nsectors_per_node * sector_len);
      // copy node coords to buffer
      memcpy(multisector_buf.get(), &h_data(cur_node_id, uint64_t(0)), ndims * sizeof(T));

      IdxT node_edges = 0;
      for (; node_edges < h_graph.extent(1); node_edges++) {
        if (h_graph(cur_node_id, node_edges) == raft::upper_bound<IdxT>()) { break; }
      }
      // write nnbrs to buffer
      nnbrs = node_edges;

      // sanity checks on nnbrs
      assert(nnbrs > 0);
      assert(nnbrs <= max_degree);

      // copy neighbors to buffer
      memcpy(nhood_buf, &h_graph(cur_node_id, 0), nnbrs * sizeof(IdxT));

      // write nnbrs to buffer
      *(IdxT*)(multisector_buf.get() + ndims * sizeof(T)) = nnbrs;

      // copy neighbors to buffer
      memcpy(multisector_buf.get() + ndims * sizeof(T) + sizeof(IdxT),
             &h_graph(cur_node_id, 0),
             nnbrs * sizeof(IdxT));

      // write sectors to disk
      output_writer.write(multisector_buf.get(), nsectors_per_node * sector_len);
    }
  }

  RAFT_LOG_DEBUG(
    "Wrote file out, index size:%lu, max_degree:%u, num_sparse:%ld, num_single:%ld, total "
    "edges:%ld, avg degree:%f",
    index_size,
    max_degree,
    num_sparse,
    num_single,
    total_edges,
    (float)total_edges / (float)h_graph.extent(0));
}

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
               bool include_dataset,
               bool sector_aligned)
{
  // Write graph to first index file (format from MSFT DiskANN OSS)
  std::ofstream index_of(file_name, std::ios::out | std::ios::binary);
  if (!index_of) { RAFT_FAIL("Cannot open file %s", file_name.c_str()); }

  auto d_graph = index_.graph();
  auto h_graph = raft::make_host_matrix<IdxT, int64_t>(d_graph.extent(0), d_graph.extent(1));
  raft::copy(h_graph.data_handle(),
             d_graph.data_handle(),
             d_graph.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // parse OPQ file if specified

  // if requested, write sector-aligned file and return
  if (sector_aligned) {
    serialize_sector_aligned<T, IdxT>(
      res, h_graph, index_.data(), static_cast<uint64_t>(index_.medoid()), index_of);
    index_of.close();
    if (!index_of) { RAFT_FAIL("Error writing output %s", file_name.c_str()); }
    return;
  }

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
