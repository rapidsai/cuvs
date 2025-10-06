/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/vamana.hpp>

#include <raft/core/error.hpp>

#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_codebooks_detail vamana deserialize_codebooks
 * @{
 */

// Parse pq_pivots file.
inline std::vector<float> parse_pq_pivots_file(const std::string& path,
                                               const int vector_dim,
                                               int32_t& pq_codebook_size,
                                               int32_t& pq_dim)
{
  std::ifstream pivots_if(path, std::ios::in | std::ios::binary);
  RAFT_EXPECTS(pivots_if, "Cannot open file %s", path.c_str());

  // check file size meets minimum for offset data
  pivots_if.ignore(std::numeric_limits<std::streamsize>::max());
  uint32_t length = pivots_if.gcount();
  RAFT_EXPECTS(length >= 40, "pq_pivots file does not contain expected metadata.");
  pivots_if.clear();  // Since ignore will have set eof.
  pivots_if.seekg(0, std::ios_base::beg);

  // check metadata
  int32_t num_offsets, num_dims;
  pivots_if.read((char*)&num_offsets, sizeof(int32_t));
  pivots_if.read((char*)&num_dims, sizeof(int32_t));
  RAFT_EXPECTS(num_offsets == 4,
               "Error reading pq_pivots file %s. # offsets = %ld; expected 4.",
               path.c_str(),
               (long)num_offsets);
  RAFT_EXPECTS(num_dims == 1,
               "Error reading pq_pivots file %s. # dimensions = %ld; expected 1.",
               path.c_str(),
               (long)num_dims);

  std::vector<int64_t> offset(num_offsets);
  pivots_if.read((char*)offset.data(), sizeof(int64_t) * num_offsets);

  // check file size meets minimum for all required data and metadata
  RAFT_EXPECTS(
    length >= offset[2] + 4,
    "pq_pivots file doesn't have the minimum expected size. Min. expected: %lld Actual: %lu",
    (long long)offset[2] + 4,
    (unsigned long)length);

  pivots_if.seekg(offset[2], std::ios_base::beg);
  pivots_if.read((char*)&pq_dim, sizeof(int32_t));
  --pq_dim;

  pivots_if.seekg(offset[0], std::ios_base::beg);
  pivots_if.read((char*)&pq_codebook_size, sizeof(int32_t));
  int32_t pq_dim_times_codebookDim;
  pivots_if.read((char*)&pq_dim_times_codebookDim, sizeof(int32_t));

  int32_t codebookDim = vector_dim / pq_dim;
  RAFT_EXPECTS(pq_dim * codebookDim == pq_dim_times_codebookDim,
               "Invalid metadata in pq_pivots file.");

  // parse pq_encoding_table
  std::vector<float> pq_encoding_table(vector_dim * pq_codebook_size);
  for (int i = 0; i < vector_dim * pq_codebook_size; i++) {
    pivots_if.read((char*)&pq_encoding_table[i],
                   4);  // Reconstruct type of the overall quantizer is int8 but because it's OPQ,
                        // need to read it as float
  }

  return pq_encoding_table;
}

// Parse rotation matrix file.
inline std::vector<float> parse_rotation_matrix_file(const std::string& path, const int vector_dim)
{
  std::ifstream mat_if(path, std::ios::in | std::ios::binary);
  if (!mat_if) { RAFT_FAIL("Cannot open file %s", path.c_str()); }

  // check file size meets minimum for metadata
  mat_if.ignore(std::numeric_limits<std::streamsize>::max());
  uint32_t length = mat_if.gcount();
  if (length < 8) { RAFT_FAIL("Rotation matrix file does not contain expected metadata."); }
  mat_if.clear();  // Since ignore will have set eof.
  mat_if.seekg(0, std::ios_base::beg);

  // check metadata
  int32_t nr, nc;
  mat_if.read((char*)&nr, sizeof(int32_t));
  mat_if.read((char*)&nc, sizeof(int32_t));
  RAFT_EXPECTS(vector_dim == nr,
               "Unexpected #rows in rotation matrix file. Expected: %ld Actual: %ld",
               (long)vector_dim,
               (long)nr);
  RAFT_EXPECTS(vector_dim == nc,
               "Unexpected #cols in rotation matrix file. Expected: %ld Actual: %ld",
               (long)vector_dim,
               (long)nc);

  // check exact length
  uint32_t length_expected = 8 + (4 * nr * nc);
  RAFT_EXPECTS(length_expected == length,
               "Rotation matrix file doesn't have expected size. Expected: %lu Actual: %ld",
               (unsigned long)length_expected,
               (long)length);

  // read rotation matrix
  std::vector<float> mat(nr * nc);
  for (int iRow = 0; iRow < nr; iRow++) {
    for (int iCol = 0; iCol < nc; iCol++) {
      mat_if.read((char*)&mat[iRow * vector_dim + iCol], 4);
    }
  }
  return mat;
}

template <typename T>
codebook_params<T> deserialize_codebooks(const std::string& codebook_prefix, const int dim)
{
  codebook_params<T> codebooks;
  codebooks.pq_encoding_table = parse_pq_pivots_file(
    codebook_prefix + "_pq_pivots.bin", dim, codebooks.pq_codebook_size, codebooks.pq_dim);
  codebooks.rotation_matrix =
    parse_rotation_matrix_file(codebook_prefix + "_pq_pivots.bin_rotation_matrix.bin", dim);
  return codebooks;
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
