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

#include <cuda.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_uvector.hpp>

#include "ann_utils.cuh"
#include <cuvs/neighbors/brute_force.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

extern "C" void run_brute_force(int64_t n_rows,
                                int64_t n_queries,
                                int64_t n_dim,
                                uint32_t n_neighbors,
                                float* index_data,
                                float* query_data,
                                uint32_t* prefilter_data,
                                float* distances_data,
                                int64_t* neighbors_data,
                                cuvsDistanceType metric);

template <typename T>
void generate_random_data(T* devPtr, size_t size)
{
  raft::handle_t handle;
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
};

template <typename index_t, typename bitmap_t = uint32_t>
index_t create_sparse_matrix(index_t m, index_t n, float sparsity, std::vector<bitmap_t>& bitmap)
{
  index_t total    = static_cast<index_t>(m * n);
  index_t num_ones = static_cast<index_t>((total * 1.0f) * sparsity);
  index_t nnz      = num_ones;

  for (auto& item : bitmap) {
    item = static_cast<bitmap_t>(0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<index_t> dis(0, total - 1);

  while (num_ones > 0) {
    index_t index = dis(gen);

    bitmap_t& element    = bitmap[index / (8 * sizeof(bitmap_t))];
    index_t bit_position = index % (8 * sizeof(bitmap_t));

    if (((element >> bit_position) & 1) == 0) {
      element |= (static_cast<bitmap_t>(1) << bit_position);
      num_ones--;
    }
  }
  return nnz;
}

template <typename index_t, typename bitmap_t = uint32_t>
void cpu_convert_to_csr(std::vector<bitmap_t>& bitmap,
                        index_t rows,
                        index_t cols,
                        std::vector<index_t>& indices,
                        std::vector<index_t>& indptr)
{
  index_t offset_indptr   = 0;
  index_t offset_values   = 0;
  indptr[offset_indptr++] = 0;

  index_t index        = 0;
  bitmap_t element     = 0;
  index_t bit_position = 0;

  for (index_t i = 0; i < rows; ++i) {
    for (index_t j = 0; j < cols; ++j) {
      index        = i * cols + j;
      element      = bitmap[index / (8 * sizeof(bitmap_t))];
      bit_position = index % (8 * sizeof(bitmap_t));

      if (((element >> bit_position) & 1)) {
        indices[offset_values] = static_cast<index_t>(j);
        offset_values++;
      }
    }
    indptr[offset_indptr++] = static_cast<index_t>(offset_values);
  }
}

template <typename value_t, typename index_t>
void cpu_sddmm(value_t* A,
               value_t* B,
               std::vector<value_t>& vals,
               const std::vector<index_t>& cols,
               const std::vector<index_t>& row_ptrs,
               bool is_row_major_A,
               bool is_row_major_B,
               index_t n_queries,
               index_t n_dataset,
               index_t dim,
               cuvsDistanceType metric,
               value_t alpha = 1.0,
               value_t beta  = 0.0)
{
  bool trans_a = is_row_major_A;
  bool trans_b = is_row_major_B;

  for (index_t i = 0; i < n_queries; ++i) {
    for (index_t j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
      value_t sum     = 0;
      value_t norms_A = 0;
      value_t norms_B = 0;
      for (index_t l = 0; l < dim; ++l) {
        index_t a_index = trans_a ? i * dim + l : l * n_queries + i;
        index_t b_index = trans_b ? l * n_dataset + cols[j] : cols[j] * dim + l;
        sum += A[a_index] * B[b_index];

        norms_A += A[a_index] * A[a_index];
        norms_B += B[b_index] * B[b_index];
      }
      vals[j] = alpha * sum + beta * vals[j];
      if (metric == cuvs::distance::DistanceType::L2Expanded) {
        vals[j] = value_t(-2.0) * vals[j] + norms_A + norms_B;
      } else if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        vals[j] = std::sqrt(value_t(-2.0) * vals[j] + norms_A + norms_B);
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        vals[j] = value_t(1.0) - vals[j] / std::sqrt(norms_A * norms_B);
      }
    }
  }
}

template <typename value_t, typename index_t>
void cpu_select_k(const std::vector<index_t>& indptr_h,
                  const std::vector<index_t>& indices_h,
                  const std::vector<value_t>& values_h,
                  std::optional<std::vector<index_t>>& in_idx_h,
                  index_t n_queries,
                  index_t n_dataset,
                  index_t n_neighbors,
                  std::vector<value_t>& out_values_h,
                  std::vector<index_t>& out_indices_h,
                  bool select_min = true)
{
  auto comp = [select_min](const std::pair<value_t, index_t>& a,
                           const std::pair<value_t, index_t>& b) {
    return select_min ? a.first < b.first : a.first >= b.first;
  };

  for (index_t row = 0; row < n_queries; ++row) {
    std::priority_queue<std::pair<value_t, index_t>,
                        std::vector<std::pair<value_t, index_t>>,
                        decltype(comp)>
      pq(comp);

    for (index_t idx = indptr_h[row]; idx < indptr_h[row + 1]; ++idx) {
      pq.push({values_h[idx], (in_idx_h.has_value()) ? (*in_idx_h)[idx] : indices_h[idx]});
      if (pq.size() > size_t(n_neighbors)) { pq.pop(); }
    }

    std::vector<std::pair<value_t, index_t>> row_pairs;
    while (!pq.empty()) {
      row_pairs.push_back(pq.top());
      pq.pop();
    }

    if (select_min) {
      std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
        return a.first <= b.first;
      });
    } else {
      std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
        return a.first >= b.first;
      });
    }
    for (index_t col = 0; col < n_neighbors; col++) {
      if (col < index_t(row_pairs.size())) {
        out_values_h[row * n_neighbors + col]  = row_pairs[col].first;
        out_indices_h[row * n_neighbors + col] = row_pairs[col].second;
      }
    }
  }
}

template <typename value_t, typename index_t, typename bitmap_t = uint32_t>
void cpu_brute_force_with_filter(value_t* query_data,
                                 value_t* index_data,
                                 std::vector<bitmap_t>& filter,
                                 std::vector<index_t>& out_indices_h,
                                 std::vector<value_t>& out_values_h,
                                 size_t n_queries,
                                 size_t n_dataset,
                                 size_t n_dim,
                                 size_t n_neighbors,
                                 size_t nnz,
                                 bool select_min,
                                 cuvsDistanceType metric)
{
  std::vector<value_t> values_h(nnz);
  std::vector<index_t> indices_h(nnz);
  std::vector<index_t> indptr_h(n_queries + 1);

  cpu_convert_to_csr(filter, (index_t)n_queries, (index_t)n_dataset, indices_h, indptr_h);

  cpu_sddmm(query_data,
            index_data,
            values_h,
            indices_h,
            indptr_h,
            true,
            false,
            (index_t)n_queries,
            (index_t)n_dataset,
            (index_t)n_dim,
            metric);

  std::optional<std::vector<index_t>> optional_indices_h = std::nullopt;

  cpu_select_k(indptr_h,
               indices_h,
               values_h,
               optional_indices_h,
               (index_t)n_queries,
               (index_t)n_dataset,
               (index_t)n_neighbors,
               out_values_h,
               out_indices_h,
               select_min);
}

template <typename T, typename IdxT>
void recall_eval(T* query_data,
                 T* index_data,
                 uint32_t* filter,
                 IdxT* neighbors,
                 T* distances,
                 size_t n_queries,
                 size_t n_rows,
                 size_t n_dim,
                 size_t n_neighbors,
                 cuvsDistanceType metric)
{
  raft::handle_t handle;
  auto distances_ref = raft::make_device_matrix<T, IdxT>(handle, n_queries, n_neighbors);
  auto neighbors_ref = raft::make_device_matrix<IdxT, IdxT>(handle, n_queries, n_neighbors);
  cuvs::neighbors::naive_knn<T, T, IdxT>(
    handle,
    distances_ref.data_handle(),
    neighbors_ref.data_handle(),
    query_data,
    index_data,
    n_queries,
    n_rows,
    n_dim,
    n_neighbors,
    static_cast<cuvs::distance::DistanceType>((uint16_t)metric));

  size_t size = n_queries * n_neighbors;
  std::vector<IdxT> neighbors_h(size);
  std::vector<T> distances_h(size);
  std::vector<IdxT> neighbors_ref_h(size);
  std::vector<T> distances_ref_h(size);

  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_h.data(), neighbors, size, stream);
  raft::copy(distances_h.data(), distances, size, stream);
  raft::copy(neighbors_ref_h.data(), neighbors_ref.data_handle(), size, stream);
  raft::copy(distances_ref_h.data(), distances_ref.data_handle(), size, stream);

  // verify output
  double min_recall = 0.95;
  ASSERT_TRUE(cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                               neighbors_h,
                                               distances_ref_h,
                                               distances_h,
                                               n_queries,
                                               n_neighbors,
                                               0.001,
                                               min_recall));
};

template <typename T, typename IdxT, typename bitmap_t = uint32_t>
void recall_eval_with_filter(T* query_data,
                             T* index_data,
                             std::vector<bitmap_t>& filter_h,
                             IdxT* neighbors_d,
                             T* distances_d,
                             std::vector<T>& distances_ref_h,
                             std::vector<IdxT>& neighbors_ref_h,
                             size_t n_queries,
                             size_t n_rows,
                             size_t n_dim,
                             uint32_t n_neighbors,
                             size_t nnz,
                             cuvsDistanceType metric)
{
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  std::vector<T> queries_h(n_queries * n_dim);
  std::vector<T> indices_h(n_rows * n_dim);

  size_t size = n_queries * n_neighbors;
  std::vector<IdxT> neighbors_h(size);
  std::vector<T> distances_h(size);

  raft::copy(neighbors_h.data(), neighbors_d, size, stream);
  raft::copy(distances_h.data(), distances_d, size, stream);
  raft::copy(queries_h.data(), query_data, n_queries * n_dim, stream);
  raft::copy(indices_h.data(), index_data, n_rows * n_dim, stream);

  bool select_min = cuvs::distance::is_min_close(metric);

  cpu_brute_force_with_filter(queries_h.data(),
                              indices_h.data(),
                              filter_h,
                              neighbors_ref_h,
                              distances_ref_h,
                              n_queries,
                              n_rows,
                              n_dim,
                              n_neighbors,
                              nnz,
                              select_min,
                              static_cast<cuvs::distance::DistanceType>((uint16_t)metric));

  // verify output
  double min_recall = 0.95;
  ASSERT_TRUE(cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                               neighbors_h,
                                               distances_ref_h,
                                               distances_h,
                                               n_queries,
                                               n_neighbors,
                                               0.001,
                                               min_recall));
};

TEST(BruteForceC, BuildSearch)
{
  int64_t n_rows       = 8096;
  int64_t n_queries    = 128;
  int64_t n_dim        = 32;
  uint32_t n_neighbors = 8;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;

  uint32_t* filter_data = NULL;

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  run_brute_force(n_rows,
                  n_queries,
                  n_dim,
                  n_neighbors,
                  index_data.data(),
                  query_data.data(),
                  filter_data,
                  distances_data.data(),
                  neighbors_data.data(),
                  metric);

  recall_eval(query_data.data(),
              index_data.data(),
              filter_data,
              neighbors_data.data(),
              distances_data.data(),
              n_queries,
              n_rows,
              n_dim,
              n_neighbors,
              metric);
}

TEST(BruteForceC, BuildSearchWithFilter)
{
  int64_t n_rows       = 8096;
  int64_t n_queries    = 128;
  int64_t n_dim        = 32;
  uint32_t n_neighbors = 8;

  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  float sparsity   = 0.2;
  int64_t n_filter = (n_queries * n_rows + 31) / 32;
  std::vector<uint32_t> filter_h(n_filter);
  int64_t nnz = create_sparse_matrix(n_queries, n_rows, sparsity, filter_h);

  cuvsDistanceType metric = L2Expanded;
  bool select_min         = cuvs::distance::is_min_close(metric);

  std::vector<float> distances_ref_h(
    n_queries * n_neighbors,
    select_min ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::lowest());
  std::vector<int64_t> neighbors_ref_h(n_queries * n_neighbors, static_cast<int64_t>(0));

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<uint32_t> filter_data(n_filter, stream);

  raft::copy(neighbors_data.data(), neighbors_ref_h.data(), n_queries * n_neighbors, stream);
  raft::copy(distances_data.data(), distances_ref_h.data(), n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  raft::copy(filter_data.data(), filter_h.data(), n_filter, stream);

  run_brute_force(n_rows,
                  n_queries,
                  n_dim,
                  n_neighbors,
                  index_data.data(),
                  query_data.data(),
                  filter_data.data(),
                  distances_data.data(),
                  neighbors_data.data(),
                  metric);

  recall_eval_with_filter(query_data.data(),
                          index_data.data(),
                          filter_h,
                          neighbors_data.data(),
                          distances_data.data(),
                          distances_ref_h,
                          neighbors_ref_h,
                          n_queries,
                          n_rows,
                          n_dim,
                          n_neighbors,
                          nnz,
                          metric);
}
