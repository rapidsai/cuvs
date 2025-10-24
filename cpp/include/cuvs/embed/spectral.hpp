/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::embed::spectral {

/**
 * Given a COO formatted (symmetric) knn graph, this function computes the spectral embeddings
 * (lowest n_components eigenvectors), using Lanczos min cut algorithm. Please note that this
 * algorithm does not compute a full laplacian eigenmap, as the laplacian eigenmap would embed each
 * connected component. Laplacian eigenmaps can be built from this algorithm by running it on the
 * vectors for each connected component.

 * @param[in] handle
 * @param[in] knn_graph KNN Graph
 * @param[in] n_components the number of components to project into
 * @param[out] out output array for embedding (size n*n_comonents)
 * @param[in] seed
 */
void fit(const raft::resources& handle,
         raft::device_coo_matrix_view<float, int, int, int> knn_graph,
         int n_components,
         raft::device_matrix_view<float, int> out,
         unsigned long long seed = 0L);
};  // namespace cuvs::embed::spectral
