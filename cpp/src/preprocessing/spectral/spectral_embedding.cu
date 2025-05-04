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

#include <cuvs/preprocessing/spectral/spectral_embedding.hpp>
#include <cuvs/preprocessing/spectral/spectral_embedding_types.hpp>
#include <raft/util/integer_utils.hpp>
// #include <cuvs/neighbors/knn.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/sparse/solver/lanczos_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <driver_types.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <cuvs/neighbors/brute_force.hpp>

#include <cstdio>
#include <iostream>

namespace cuvs::preprocessing::spectral {

auto spectral_embedding(
  raft::resources const& handle,
  raft::device_matrix_view<float, int, raft::row_major> nums,
  raft::device_matrix_view<float, int, raft::col_major> embedding,
  cuvs::preprocessing::spectral::spectral_embedding_config spectral_embedding_config) -> int
{
  // Define our sample data (similar to the Python example)
  const int n_samples     = nums.extent(0);
  const int n_features    = nums.extent(1);
  const int k             = spectral_embedding_config.n_neighbors;  // Number of neighbors
  const bool include_self = false;  // Set to false to exclude self-connections
  const bool drop_first   = spectral_embedding_config.drop_first;

  auto stream = raft::resource::get_cuda_stream(handle);
  // raft::device_resources res(stream);

  // If not including self, we need to request k+1 neighbors
  int k_search = include_self ? k : k + 1;

  cuvs::neighbors::brute_force::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2SqrtExpanded;

  auto d_indices   = raft::make_device_matrix<int64_t>(handle, n_samples, k_search);
  auto d_distances = raft::make_device_matrix<float>(handle, n_samples, k_search);

  auto index =
    cuvs::neighbors::brute_force::build(handle, index_params, raft::make_const_mdspan(nums));

  cuvs::neighbors::brute_force::search_params search_params;

  cuvs::neighbors::brute_force::search(
    handle, search_params, index, nums, d_indices.view(), d_distances.view());

  // Create a COO matrix for the KNN graph
  raft::sparse::COO<float> knn_coo(stream, n_samples, n_samples);

  // Resize COO to actual nnz
  size_t nnz = n_samples * k_search;
  knn_coo.allocate(nnz, n_samples, false, stream);

  auto knn_rows = raft::make_device_vector<int>(handle, nnz);
  auto knn_cols = raft::make_device_vector<int>(handle, nnz);

  thrust::transform(thrust::device,
                    d_indices.data_handle(),
                    d_indices.data_handle() + nnz,
                    knn_cols.data_handle(),
                    [] __device__(int64_t x) -> int { return static_cast<int>(x); });

  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(nnz),
                    knn_rows.data_handle(),
                    [=] __device__(int64_t i) { return static_cast<int>(i / k_search); });

  // Copy COO data to device
  raft::copy(knn_coo.rows(), knn_rows.data_handle(), nnz, stream);
  raft::copy(knn_coo.cols(), knn_cols.data_handle(), nnz, stream);
  raft::copy(knn_coo.vals(), d_distances.data_handle(), nnz, stream);

  raft::sparse::COO<float> coo_no_zeros(stream);  // Don't pre-allocate dimensions
  raft::sparse::op::coo_remove_zeros<float>(&knn_coo, &coo_no_zeros, stream);

  // binarize to 1s
  thrust::fill(thrust::device, coo_no_zeros.vals(), coo_no_zeros.vals() + coo_no_zeros.nnz, 1.0f);

  // Create output COO for symmetrized result - create unallocated COO
  raft::sparse::COO<float> sym_coo1(stream);  // Don't pre-allocate dimensions

  // // Define the reduction function with the correct signature
  auto reduction_op = [] __device__(int row, int col, float a, float b) {
    // Only use the values, ignore row/col indices
    return 0.5f * (a + b);
  };

  // Symmetrize the matrix
  raft::sparse::linalg::coo_symmetrize(&coo_no_zeros, &sym_coo1, reduction_op, stream);

  raft::sparse::op::coo_sort<float>(&sym_coo1, stream);

  raft::sparse::COO<float> sym_coo(stream);  // Don't pre-allocate dimensions
  raft::sparse::op::coo_remove_zeros<float>(&sym_coo1, &sym_coo, stream);

  nnz = sym_coo.nnz;
  printf("\nSymmetrized COO Matrix (nnz=%ld):\n", nnz);

  using value_idx = int;
  using value_t   = float;
  using size_type = size_t;

  raft::sparse::op::coo_sort<float>(&sym_coo, stream);
  auto row_ind = raft::make_device_vector<int>(handle, sym_coo.n_rows + 1);
  raft::sparse::convert::sorted_coo_to_csr(&sym_coo, row_ind.data_handle(), stream);

  const int one = sym_coo.nnz;
  raft::copy(row_ind.data_handle() + row_ind.size() - 1, &one, 1, stream);

  auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
    const_cast<int*>(row_ind.data_handle()),
    const_cast<int*>(sym_coo.cols()),
    sym_coo.n_rows,
    sym_coo.n_cols,
    sym_coo.nnz);

  auto csr_matrix_view = raft::make_device_csr_matrix_view<float, int, int, int>(
    const_cast<float*>(sym_coo.vals()), csr_structure);

  auto diagonal            = raft::make_device_vector<float>(handle, csr_structure.get_n_rows());
  auto laplacian           = spectral_embedding_config.norm_laplacian
                               ? raft::sparse::linalg::compute_graph_laplacian_normalized(
                         handle, csr_matrix_view, diagonal.view())
                               : raft::sparse::linalg::compute_graph_laplacian(handle, csr_matrix_view);
  auto laplacian_structure = laplacian.structure_view();

  // L *= -1
  thrust::transform(thrust::device,
                    laplacian.get_elements().data(),
                    laplacian.get_elements().data() + laplacian_structure.get_nnz(),
                    laplacian.get_elements().data(),
                    [] __device__(float x) { return -x; });

  auto config           = raft::sparse::solver::lanczos_solver_config<float>();
  config.n_components   = spectral_embedding_config.n_components;
  config.max_iterations = 1000;
  config.ncv =
    std::min(laplacian_structure.get_n_rows(), std::max(2 * config.n_components + 1, 20));
  config.tolerance = 1e-5;
  config.which     = raft::sparse::solver::LANCZOS_WHICH::LA;
  config.seed      = spectral_embedding_config.seed;

  auto eigenvalues =
    raft::make_device_vector<float, int, raft::col_major>(handle, config.n_components);
  auto eigenvectors = raft::make_device_matrix<float, int, raft::col_major>(
    handle, laplacian_structure.get_n_rows(), config.n_components);

  raft::sparse::solver::lanczos_compute_smallest_eigenvectors<int, float>(
    handle,
    config,
    raft::make_device_csr_matrix_view<float, int, int, int>(laplacian.get_elements().data(),
                                                            laplacian_structure),
    std::nullopt,
    eigenvalues.view(),
    eigenvectors.view());

  raft::print_device_vector(
    "eigenvalues", eigenvalues.data_handle(), eigenvalues.size(), std::cout);

  if (spectral_embedding_config.norm_laplacian) {
    raft::linalg::matrix_vector_op(
      handle,
      raft::make_const_mdspan(eigenvectors.view()),  // input matrix view
      raft::make_const_mdspan(diagonal.view()),      // input vector view
      eigenvectors.view(),                           // output matrix view (in-place)
      raft::linalg::Apply::ALONG_COLUMNS,  // divide each row by corresponding diagonal element
      [] __device__(float elem, float diag) { return elem / diag; });
  }

  // Replace the direct copy with a gather operation that reverses columns

  // Create a sequence of reversed column indices
  config.n_components = drop_first ? config.n_components - 1 : config.n_components;
  auto col_indices    = raft::make_device_vector<int>(handle, config.n_components);
  thrust::sequence(thrust::device,
                   col_indices.data_handle(),
                   col_indices.data_handle() + config.n_components,
                   config.n_components - 1,  // Start from the last column index
                   -1                        // Decrement (move backward)
  );

  // Create row-major views of the column-major matrices
  // This is just a view re-interpretation, no data movement
  auto eigenvectors_row_view = raft::make_device_matrix_view<float, int, raft::row_major>(
    eigenvectors.data_handle(),
    eigenvectors.extent(1),  // Swap dimensions for the view
    eigenvectors.extent(0));

  auto embedding_row_view = raft::make_device_matrix_view<float, int, raft::row_major>(
    embedding.data_handle(),
    embedding.extent(1),  // Swap dimensions for the view
    embedding.extent(0));

  raft::matrix::gather<float, int, int>(
    handle,
    raft::make_const_mdspan(eigenvectors_row_view),  // Source matrix (as row-major view)
    raft::make_const_mdspan(col_indices.view()),     // Column indices to gather
    embedding_row_view                               // Destination matrix (as row-major view)
  );

  return 0;
}

}  // namespace cuvs::preprocessing::spectral
