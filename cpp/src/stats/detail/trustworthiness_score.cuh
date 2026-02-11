/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/col_wise_sort.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#define N_THREADS 512

namespace cuvs::stats::detail {

/**
 * @brief Build the lookup table
 * @param[out] lookup_table: Lookup table giving nearest neighbor order
 *                of pairwise distance calculations given sample index
 * @param[in] x_ind: Sorted indexes of pairwise distance calculations of X
 * @param n: Number of samples
 * @param work: Number of elements to consider
 */
RAFT_KERNEL build_lookup_table(int* lookup_table, const int* x_ind, int n, int work)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= work) return;

  int sample_idx = i / n;
  int nn_idx     = i % n;

  int idx                              = x_ind[i];
  lookup_table[(sample_idx * n) + idx] = nn_idx;
}

/**
 * @brief Compute a the rank of trustworthiness score
 * @param[out] rank: Resulting rank
 * @param[out] lookup_table: Lookup table giving nearest neighbor order
 *                of pairwise distance calculations given sample index
 * @param[in] emb_ind: Indexes of KNN on embeddings
 * @param n: Number of samples
 * @param n_neighbors: Number of neighbors considered by trustworthiness score
 * @param work: Batch to consider (to do it at once use n * n_neighbors)
 */
template <typename KnnIndexT>
RAFT_KERNEL compute_rank(
  double* rank, const int* lookup_table, const KnnIndexT* emb_ind, int n, int n_neighbors, int work)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= work) return;

  int sample_idx = i / n_neighbors;

  KnnIndexT emb_nn_ind = emb_ind[i];

  int r   = lookup_table[(sample_idx * n) + emb_nn_ind];
  int tmp = r - n_neighbors + 1;
  if (tmp > 0) raft::myAtomicAdd<double>(rank, tmp);
}

/**
 * @brief Compute a kNN and returns the indices of the nearest neighbors
 * @param h Raft handle
 * @param[in] input Input matrix containing the dataset
 * @param n Number of samples
 * @param d Number of features
 * @param n_neighbors number of neighbors
 * @param[out] indices KNN indexes
 * @param[out] distances KNN distances
 */
template <typename MathT>
void run_knn(const raft::resources& h,
             MathT* input,
             cuvs::distance::DistanceType metric,
             int n,
             int d,
             int n_neighbors,
             int64_t* indices,
             MathT* distances)
{
  auto input_view = raft::make_device_matrix_view<const MathT, int64_t>(input, n, d);
  auto index      = cuvs::neighbors::brute_force::build(h, input_view, metric);

  cuvs::neighbors::brute_force::search(
    h,
    index,
    input_view,
    raft::make_device_matrix_view<int64_t, int64_t>(indices, n, n_neighbors),
    raft::make_device_matrix_view<MathT, int64_t>(distances, n, n_neighbors),
    cuvs::neighbors::filtering::none_sample_filter{});
}

/**
 * @brief Compute the trustworthiness score
 * @param h Raft handle
 * @param X[in]: Data in original dimension
 * @param X_embedded[in]: Data in target dimension (embedding)
 * @param n: Number of samples
 * @param m: Number of features in high/original dimension
 * @param d: Number of features in low/embedded dimension
 * @param n_neighbors Number of neighbors considered by trustworthiness score
 * @param batchSize Batch size
 * @return Trustworthiness score
 */
template <typename MathT>
auto trustworthiness_score(const raft::resources& h,
                           const MathT* X,
                           MathT* X_embedded,
                           cuvs::distance::DistanceType metric,
                           int n,
                           int m,
                           int d,
                           int n_neighbors,
                           int batchSize = 512) -> double
{
  cudaStream_t stream = raft::resource::get_cuda_stream(h);

  const int knn_alloc = n * (n_neighbors + 1);
  rmm::device_uvector<int64_t> emb_ind(knn_alloc, stream);
  rmm::device_uvector<MathT> emb_dist(knn_alloc, stream);

  run_knn(h, X_embedded, metric, n, d, n_neighbors + 1, emb_ind.data(), emb_dist.data());

  const int pairwise_alloc = batchSize * n;
  rmm::device_uvector<int> x_ind(pairwise_alloc, stream);
  rmm::device_uvector<MathT> x_dist(pairwise_alloc, stream);
  rmm::device_uvector<int> lookup_table(pairwise_alloc, stream);

  double t = 0.0;
  rmm::device_scalar<double> t_dbuf(stream);

  int to_do = n;
  while (to_do > 0) {
    int cur_batch_size = min(to_do, batchSize);

    // Takes at most batchSize vectors at a time
    cuvs::distance::pairwise_distance(
      h,
      raft::make_device_matrix_view<const float, std::int64_t>(
        &X[(n - to_do) * m], cur_batch_size, m),
      raft::make_device_matrix_view<const float, std::int64_t>(X, n, m),
      raft::make_device_matrix_view<float, std::int64_t>(x_dist.data(), cur_batch_size, n),
      metric);

    size_t col_sort_workspace_size = 0;
    bool b_alloc_workspace         = false;

    raft::matrix::sort_cols_per_row(x_dist.data(),
                                    x_ind.data(),
                                    cur_batch_size,
                                    n,
                                    b_alloc_workspace,
                                    nullptr,
                                    col_sort_workspace_size,
                                    stream);

    if (b_alloc_workspace) {
      rmm::device_uvector<char> sort_cols_workspace(col_sort_workspace_size, stream);

      raft::matrix::sort_cols_per_row(x_dist.data(),
                                      x_ind.data(),
                                      cur_batch_size,
                                      n,
                                      b_alloc_workspace,
                                      sort_cols_workspace.data(),
                                      col_sort_workspace_size,
                                      stream);
    }

    int work     = cur_batch_size * n;
    int n_blocks = raft::ceildiv(work, N_THREADS);
    build_lookup_table<<<n_blocks, N_THREADS, 0, stream>>>(
      lookup_table.data(), x_ind.data(), n, work);

    RAFT_CUDA_TRY(cudaMemsetAsync(t_dbuf.data(), 0, sizeof(double), stream));

    work     = cur_batch_size * (n_neighbors + 1);
    n_blocks = raft::ceildiv(work, N_THREADS);
    compute_rank<<<n_blocks, N_THREADS, 0, stream>>>(
      t_dbuf.data(),
      lookup_table.data(),
      &emb_ind.data()[(n - to_do) * (n_neighbors + 1)],
      n,
      n_neighbors + 1,
      work);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    t += t_dbuf.value(stream);

    to_do -= cur_batch_size;
  }

  t = 1.0 - ((2.0 / ((n * n_neighbors) * ((2.0 * n) - (3.0 * n_neighbors) - 1.0))) * t);

  return t;
}

}  // namespace cuvs::stats::detail
