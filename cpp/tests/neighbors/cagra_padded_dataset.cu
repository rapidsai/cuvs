/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Tests that CAGRA build and search work with device_padded_dataset and
 * device_padded_dataset_view. Includes the CAGRA implementation so the test
 * binary provides the padded build overload symbols regardless of which
 * libcuvs is loaded at runtime.
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
// Padded dataset view: build CAGRA from device_padded_dataset_view, search, check recall
// ---------------------------------------------------------------------------
TEST(CagraPaddedDataset, PaddedDatasetViewBuildSearchRecall)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  const int64_t n_rows    = 500;
  const uint32_t dim      = 32;
  const int64_t n_queries = 50;
  const uint32_t k        = 16;

  rmm::device_uvector<float> database(n_rows * dim, stream);
  rmm::device_uvector<float> queries(n_queries * dim, stream);
  raft::random::RngState r(12345ULL);
  raft::random::normal(res, r, database.data(), n_rows * dim, 0.0f, 1.0f);
  raft::random::normal(res, r, queries.data(), n_queries * dim, 0.0f, 1.0f);
  raft::resource::sync_stream(res);

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

  cagra::index_params build_params;
  build_params.metric             = cuvs::distance::DistanceType::L2Expanded;
  build_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
    raft::matrix_extent<int64_t>(n_rows, dim), build_params.metric);

  // Build from device_padded_dataset_view (dim=32 -> stride=32 is valid for alignment)
  auto db_view = raft::make_device_matrix_view<const float, int64_t>(database.data(), n_rows, dim);
  auto padded_view = cuvs::neighbors::make_padded_dataset_view(res, db_view);
  cagra::index<float, uint32_t> index = cagra::build(res, build_params, padded_view);

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

  const double min_recall = 0.9;
  EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indices_naive,
                                               indices_cagra,
                                               distances_naive,
                                               distances_cagra,
                                               n_queries,
                                               k,
                                               0.003,
                                               min_recall));
}

// ---------------------------------------------------------------------------
// Padded dataset (owning): build CAGRA from device_padded_dataset (move), search, check recall
// ---------------------------------------------------------------------------
TEST(CagraPaddedDataset, PaddedDatasetBuildSearchRecall)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  const int64_t n_rows    = 500;
  const uint32_t dim      = 32;
  const int64_t n_queries = 50;
  const uint32_t k        = 16;

  rmm::device_uvector<float> database(n_rows * dim, stream);
  rmm::device_uvector<float> queries(n_queries * dim, stream);
  raft::random::RngState r(54321ULL);
  raft::random::normal(res, r, database.data(), n_rows * dim, 0.0f, 1.0f);
  raft::random::normal(res, r, queries.data(), n_queries * dim, 0.0f, 1.0f);
  raft::resource::sync_stream(res);

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

  // Owning device padded dataset: allocate with correct stride, copy, then build from view.
  // (First test uses make_padded_dataset_view for non-owning; here we own the buffer.)
  auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  raft::copy(dev_matrix.data_handle(), database.data(), static_cast<size_t>(n_rows * dim), stream);
  raft::resource::sync_stream(res);
  auto ds = std::make_unique<cuvs::neighbors::device_padded_dataset<float, int64_t>>(
    std::move(dev_matrix), dim);

  cagra::index_params build_params;
  build_params.metric             = cuvs::distance::DistanceType::L2Expanded;
  build_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
    raft::matrix_extent<int64_t>(n_rows, dim), build_params.metric);

  cagra::index<float, uint32_t> index = cagra::build(res, build_params, ds->as_dataset_view());

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

  const double min_recall = 0.9;
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
