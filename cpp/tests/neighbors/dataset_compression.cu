/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tests that exercise real compression (codebook training + encoding) and verify
 * correctness by comparing search results on the compressed dataset to ground truth
 * from brute-force search on the raw vectors.
 *
 * This is Option A: build with VPQ (codebook training + encoding), run search on the
 * compressed dataset, then compare recall to brute-force KNN on the raw vectors.
 * The CAGRA parameterized tests (ann_cagra.cuh with compression = vpq_params) do the
 * same thing; this test is a single, focused case that lives alongside the dataset API
 * tests (dataset_types.cu) so compression correctness is easy to find and run.
 */

#include "ann_utils.cuh"
#include "naive_knn.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::test {

using namespace cuvs::neighbors::cagra;

// ---------------------------------------------------------------------------
// VPQ compression: build CAGRA with VPQ, search, compare recall to naive on raw
// ---------------------------------------------------------------------------
TEST(DatasetCompression, VpqBuildSearchRecall)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  const int64_t n_rows    = 500;
  const uint32_t dim      = 32;
  const int64_t n_queries = 50;
  const uint32_t k        = 16;

  // 1. Generate data (same idea as CAGRA tests: small random dataset)
  rmm::device_uvector<float> database(n_rows * dim, stream);
  rmm::device_uvector<float> queries(n_queries * dim, stream);
  raft::random::RngState r(12345ULL);
  raft::random::normal(res, r, database.data(), n_rows * dim, 0.0f, 1.0f);
  raft::random::normal(res, r, queries.data(), n_queries * dim, 0.0f, 1.0f);
  raft::resource::sync_stream(res);

  // 2. Ground truth: brute-force KNN on raw vectors
  const size_t queries_size = n_queries * k;
  rmm::device_uvector<float> distances_naive_dev(queries_size, stream);
  rmm::device_uvector<int64_t> indices_naive_dev(queries_size, stream);
  cuvs::neighbors::naive_knn<float, float, int64_t>(res,
                                                    distances_naive_dev.data(),
                                                    indices_naive_dev.data(),
                                                    queries.data(),
                                                    database.data(),
                                                    n_queries,
                                                    n_rows,
                                                    dim,
                                                    k,
                                                    cuvs::distance::DistanceType::L2Expanded);
  std::vector<float> distances_naive(queries_size);
  std::vector<int64_t> indices_naive(queries_size);
  raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream);
  raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream);
  raft::resource::sync_stream(res);

  // 3. Build CAGRA with VPQ compression (trains codebooks, encodes data, index holds vpq_dataset)
  cagra::index_params build_params;
  build_params.metric             = cuvs::distance::DistanceType::L2Expanded;
  build_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
    raft::matrix_extent<int64_t>(n_rows, dim), build_params.metric);
  build_params.compression               = cuvs::neighbors::vpq_params{};
  build_params.compression->pq_bits      = 8;
  build_params.compression->pq_dim       = dim / 2;  // 16 subspaces of length 2
  build_params.compression->vq_n_centers = 64;

  auto database_view =
    raft::make_device_matrix_view<const float, int64_t>(database.data(), n_rows, dim);
  cagra::index<float, uint32_t> index = cagra::build(res, build_params, database_view);

  // 4. Search on the compressed index (uses vpq_dataset for distance computation)
  rmm::device_uvector<float> distances_cagra_dev(queries_size, stream);
  rmm::device_uvector<int64_t> indices_cagra_dev(queries_size, stream);
  cagra::search_params sp;
  sp.algo = cagra::search_algo::AUTO;

  auto queries_view =
    raft::make_device_matrix_view<const float, int64_t>(queries.data(), n_queries, dim);
  auto indices_out_view =
    raft::make_device_matrix_view<int64_t, int64_t>(indices_cagra_dev.data(), n_queries, k);
  auto dists_out_view =
    raft::make_device_matrix_view<float, int64_t>(distances_cagra_dev.data(), n_queries, k);

  cagra::search(res, sp, index, queries_view, indices_out_view, dists_out_view);

  std::vector<float> distances_cagra(queries_size);
  std::vector<int64_t> indices_cagra(queries_size);
  raft::update_host(distances_cagra.data(), distances_cagra_dev.data(), queries_size, stream);
  raft::update_host(indices_cagra.data(), indices_cagra_dev.data(), queries_size, stream);
  raft::resource::sync_stream(res);

  // 5. Compare recall (compressed search vs ground truth on raw)
  // VPQ is lossy so we use a relaxed min_recall (e.g. 0.5); CAGRA parameterized tests use ~0.6
  const double min_recall = 0.5;
  EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indices_naive,
                                               indices_cagra,
                                               distances_naive,
                                               distances_cagra,
                                               n_queries,
                                               k,
                                               0.003,
                                               min_recall));
}

}  // namespace cuvs::neighbors::test
