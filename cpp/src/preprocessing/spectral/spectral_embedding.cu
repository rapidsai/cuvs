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

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/sparse/solver/lanczos_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <thrust/sequence.h>
#include <thrust/tabulate.h>

namespace cuvs::preprocessing::spectral_embedding {

void create_connectivity_graph(raft::resources const& handle,
                               params spectral_embedding_config,
                               raft::device_matrix_view<float, int, raft::row_major> dataset,
                               raft::device_matrix_view<float, int, raft::col_major> embedding,
                               raft::device_coo_matrix<float, int, int, int>& connectivity_graph)
{
  const int n_samples  = dataset.extent(0);
  const int n_features = dataset.extent(1);
  const int k_search   = spectral_embedding_config.n_neighbors;
  const size_t nnz     = n_samples * k_search;

  auto stream = raft::resource::get_cuda_stream(handle);

  cuvs::neighbors::brute_force::search_params search_params;
  cuvs::neighbors::brute_force::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2SqrtExpanded;

  auto d_indices   = raft::make_device_matrix<int64_t>(handle, n_samples, k_search);
  auto d_distances = raft::make_device_matrix<float>(handle, n_samples, k_search);

  auto index =
    cuvs::neighbors::brute_force::build(handle, index_params, raft::make_const_mdspan(dataset));

  cuvs::neighbors::brute_force::search(
    handle, search_params, index, dataset, d_indices.view(), d_distances.view());

  auto knn_rows = raft::make_device_vector<int>(handle, nnz);
  auto knn_cols = raft::make_device_vector<int>(handle, nnz);

  raft::linalg::unary_op(
    handle, make_const_mdspan(d_indices.view()), knn_cols.view(), [] __device__(int64_t x) {
      return static_cast<int>(x);
    });

  thrust::tabulate(raft::resource::get_thrust_policy(handle),
                   knn_rows.data_handle(),
                   knn_rows.data_handle() + nnz,
                   [k_search] __device__(int idx) { return idx / k_search; });

  // set all distances to 1.0f (connectivity KNN graph)
  raft::matrix::fill(handle, raft::make_device_vector_view(d_distances.data_handle(), nnz), 1.0f);

  auto coo_matrix_view = raft::make_device_coo_matrix_view<const float, int, int, int>(
    d_distances.data_handle(),
    raft::make_device_coordinate_structure_view<int, int, int>(
      knn_rows.data_handle(), knn_cols.data_handle(), n_samples, n_samples, nnz));

  auto sym_coo1_matrix =
    raft::make_device_coo_matrix<float, int, int, int>(handle, n_samples, n_samples);
  raft::sparse::linalg::coo_symmetrize<128, float, int, int>(
    handle, coo_matrix_view, sym_coo1_matrix, [] __device__(int row, int col, float a, float b) {
      return 0.5f * (a + b);
    });

  raft::sparse::op::coo_sort<float>(n_samples,
                                    n_samples,
                                    sym_coo1_matrix.structure_view().get_nnz(),
                                    sym_coo1_matrix.structure_view().get_rows().data(),
                                    sym_coo1_matrix.structure_view().get_cols().data(),
                                    sym_coo1_matrix.get_elements().data(),
                                    stream);

  raft::sparse::op::coo_remove_scalar<128, float, int, int>(
    handle,
    raft::make_device_coo_matrix_view<const float, int, int, int>(
      sym_coo1_matrix.get_elements().data(), sym_coo1_matrix.structure_view()),
    raft::make_host_scalar<float>(0.0f).view(),
    connectivity_graph);
}

raft::device_csr_matrix_view<float, int, int, int> coo_to_csr_matrix(
  raft::resources const& handle,
  const int n_samples,
  raft::device_vector_view<int> sym_coo_row_ind,
  raft::device_coo_matrix_view<float, int, int, int> sym_coo_matrix_view)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  raft::sparse::op::coo_sort<float>(n_samples,
                                    n_samples,
                                    sym_coo_matrix_view.structure_view().get_nnz(),
                                    sym_coo_matrix_view.structure_view().get_rows().data(),
                                    sym_coo_matrix_view.structure_view().get_cols().data(),
                                    sym_coo_matrix_view.get_elements().data(),
                                    stream);

  raft::sparse::convert::sorted_coo_to_csr(sym_coo_matrix_view.structure_view().get_rows().data(),
                                           sym_coo_matrix_view.structure_view().get_nnz(),
                                           sym_coo_row_ind.data_handle(),
                                           n_samples,
                                           stream);

  auto sym_coo_nnz = sym_coo_matrix_view.structure_view().get_nnz();
  raft::copy(sym_coo_row_ind.data_handle() + sym_coo_row_ind.size() - 1, &sym_coo_nnz, 1, stream);

  auto csr_matrix_view = raft::make_device_csr_matrix_view<float, int, int, int>(
    const_cast<float*>(sym_coo_matrix_view.get_elements().data()),
    raft::make_device_compressed_structure_view<int, int, int>(
      const_cast<int*>(sym_coo_row_ind.data_handle()),
      const_cast<int*>(sym_coo_matrix_view.structure_view().get_cols().data()),
      n_samples,
      n_samples,
      sym_coo_matrix_view.structure_view().get_nnz()));
  return csr_matrix_view;
}

raft::device_csr_matrix<float, int, int, int> create_laplacian(
  raft::resources const& handle,
  params spectral_embedding_config,
  raft::device_csr_matrix_view<float, int, int, int> csr_matrix_view,
  raft::device_vector_view<float, int> diagonal)
{
  auto laplacian = spectral_embedding_config.norm_laplacian
                     ? raft::sparse::linalg::laplacian_normalized(handle, csr_matrix_view, diagonal)
                     : raft::sparse::linalg::compute_graph_laplacian(handle, csr_matrix_view);

  auto laplacian_elements_view = raft::make_device_vector_view<float, int>(
    laplacian.get_elements().data(), laplacian.structure_view().get_nnz());

  raft::linalg::unary_op(handle,
                         raft::make_const_mdspan(laplacian_elements_view),
                         laplacian_elements_view,
                         [] __device__(float x) { return -x; });

  return laplacian;
}

void compute_eigenpairs(raft::resources const& handle,
                        params spectral_embedding_config,
                        const int n_samples,
                        raft::device_csr_matrix<float, int, int, int> laplacian,
                        raft::device_vector_view<float, int> diagonal,
                        raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  auto config           = raft::sparse::solver::lanczos_solver_config<float>();
  config.n_components   = spectral_embedding_config.n_components;
  config.max_iterations = 1000;
  config.ncv            = std::min(n_samples, std::max(2 * config.n_components + 1, 20));
  config.tolerance      = 1e-5;
  config.which          = raft::sparse::solver::LANCZOS_WHICH::LA;
  config.seed           = spectral_embedding_config.seed;

  auto eigenvalues =
    raft::make_device_vector<float, int, raft::col_major>(handle, config.n_components);
  auto eigenvectors =
    raft::make_device_matrix<float, int, raft::col_major>(handle, n_samples, config.n_components);

  raft::sparse::solver::lanczos_compute_smallest_eigenvectors<int, float>(
    handle,
    config,
    raft::make_device_csr_matrix_view<float, int, int, int>(laplacian.get_elements().data(),
                                                            laplacian.structure_view()),
    std::nullopt,
    eigenvalues.view(),
    eigenvectors.view());

  if (spectral_embedding_config.norm_laplacian) {
    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_const_mdspan(eigenvectors.view()),  // input matrix view
      raft::make_const_mdspan(diagonal),             // input vector view
      eigenvectors.view(),                           // output matrix view (in-place)
      [] __device__(float elem, float diag) { return elem / diag; });
  }

  // Create a sequence of reversed column indices
  config.n_components =
    spectral_embedding_config.drop_first ? config.n_components - 1 : config.n_components;
  auto col_indices = raft::make_device_vector<int>(handle, config.n_components);

  // TODO: https://github.com/rapidsai/raft/issues/2661
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
}

void transform(raft::resources const& handle,
               params spectral_embedding_config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  const int n_samples = dataset.extent(0);

  auto sym_coo_matrix =
    raft::make_device_coo_matrix<float, int, int, int>(handle, n_samples, n_samples);
  auto sym_coo_row_ind = raft::make_device_vector<int>(handle, n_samples + 1);
  auto diagonal        = raft::make_device_vector<float, int>(handle, n_samples);

  create_connectivity_graph(handle, spectral_embedding_config, dataset, embedding, sym_coo_matrix);
  auto csr_matrix_view =
    coo_to_csr_matrix(handle, n_samples, sym_coo_row_ind.view(), sym_coo_matrix.view());
  auto laplacian =
    create_laplacian(handle, spectral_embedding_config, csr_matrix_view, diagonal.view());
  compute_eigenpairs(
    handle, spectral_embedding_config, n_samples, laplacian, diagonal.view(), embedding);
}

void transform(raft::resources const& handle,
               params spectral_embedding_config,
               raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  const int n_samples = connectivity_graph.structure_view().get_n_rows();

  auto sym_coo_row_ind = raft::make_device_vector<int>(handle, n_samples + 1);
  auto diagonal        = raft::make_device_vector<float, int>(handle, n_samples);

  auto csr_matrix_view =
    coo_to_csr_matrix(handle, n_samples, sym_coo_row_ind.view(), connectivity_graph);
  auto laplacian =
    create_laplacian(handle, spectral_embedding_config, csr_matrix_view, diagonal.view());
  compute_eigenpairs(
    handle, spectral_embedding_config, n_samples, laplacian, diagonal.view(), embedding);
}

}  // namespace cuvs::preprocessing::spectral_embedding
