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

#include "../sparse/cluster/detail/spectral.cuh"
#include <cuvs/embed/spectral.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::embed::spectral {

/**
 * Given a COO formatted (symmetric) knn graph, this function computes the spectral embeddings
 * (lowest n_components eigenvectors), using Lanczos min cut algorithm.
 * @param rows source vertices of knn graph (size nnz)
 * @param cols destination vertices of knn graph (size nnz)
 * @param vals edge weights connecting vertices of knn graph (size nnz)
 * @param nnz size of rows/cols/vals
 * @param n number of samples in X
 * @param n_neighbors the number of neighbors to query for knn graph construction
 * @param n_components the number of components to project the X into
 * @param out output array for embedding (size n*n_comonents)
 */
void fit(const raft::resources& handle,
         raft::device_coo_matrix_view<float, int, int, int> knn_graph,
         int n_components,
         raft::device_matrix_view<float, int> out,
         unsigned long long seed)
{
  cuvs::sparse::cluster::spectral::detail::fit_embedding(
    handle,
    knn_graph.structure_view().get_rows().data(),
    knn_graph.structure_view().get_cols().data(),
    knn_graph.get_elements().data(),
    knn_graph.structure_view().get_nnz(),
    knn_graph.structure_view().get_n_rows(),
    n_components,
    out.data_handle(),
    seed);
}
};  // namespace cuvs::embed::spectral
