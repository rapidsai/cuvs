/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Tests that CAGRA build only attaches a view to the index (never takes ownership).
 * After build, index.data().is_owning() must be false. This documents the invariant
 * that build is migrated to view-only; update/merge/extend may still pass ownership
 * via update_dataset(unique_ptr&&).
 */

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
// Build from device_padded_dataset_view (non-owning view): index must not own.
// ---------------------------------------------------------------------------
TEST(CagraBuildViewOnly, BuildFromViewIndexDoesNotOwn)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  const int64_t n_rows = 200;
  const uint32_t dim   = 16;

  rmm::device_uvector<float> database(n_rows * dim, stream);
  raft::random::RngState r(12345ULL);
  raft::random::normal(res, r, database.data(), n_rows * dim, 0.0f, 1.0f);
  raft::resource::sync_stream(res);

  cagra::index_params build_params;
  build_params.metric = cuvs::distance::DistanceType::L2Expanded;
  build_params.graph_build_params =
    cagra::graph_build_params::ivf_pq_params(raft::matrix_extent<int64_t>(n_rows, dim), build_params.metric);

  auto db_view = raft::make_device_matrix_view<const float, int64_t>(database.data(), n_rows, dim);
  auto padded_view = cuvs::neighbors::make_padded_dataset_view(res, db_view);

  cagra::index<float, uint32_t> index = cagra::build(res, build_params, padded_view);

  // Build only takes a view; index must not own the dataset.
  EXPECT_FALSE(index.data().is_owning())
    << "Build must attach only a view; index must not own the dataset.";
}

// ---------------------------------------------------------------------------
// Build from owning device_padded_dataset via .as_dataset_view(): index must not own.
// Caller owns the buffer and passes a view; index must still hold only a view.
// ---------------------------------------------------------------------------
TEST(CagraBuildViewOnly, BuildFromOwnedDatasetViaViewIndexDoesNotOwn)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  const int64_t n_rows = 200;
  const uint32_t dim   = 16;

  rmm::device_uvector<float> database(n_rows * dim, stream);
  raft::random::RngState r(54321ULL);
  raft::random::normal(res, r, database.data(), n_rows * dim, 0.0f, 1.0f);
  raft::resource::sync_stream(res);

  auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  raft::copy(dev_matrix.data_handle(), database.data(), static_cast<size_t>(n_rows * dim), stream);
  raft::resource::sync_stream(res);

  auto ds = std::make_unique<cuvs::neighbors::device_padded_dataset<float, int64_t>>(
    std::move(dev_matrix), dim);

  cagra::index_params build_params;
  build_params.metric = cuvs::distance::DistanceType::L2Expanded;
  build_params.graph_build_params =
    cagra::graph_build_params::ivf_pq_params(raft::matrix_extent<int64_t>(n_rows, dim), build_params.metric);

  // Pass view only; caller keeps ds for lifetime of index.
  cagra::index<float, uint32_t> index = cagra::build(res, build_params, ds->as_dataset_view());

  // Index must hold only the view, not take ownership of ds.
  EXPECT_FALSE(index.data().is_owning())
    << "Build must attach only a view even when caller has an owning dataset; "
    << "index must not own the dataset.";
}

}  // namespace cuvs::neighbors::test
