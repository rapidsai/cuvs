/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

#include <raft/random/rng.cuh>

#include <cstring>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::ivf_pq {

using f32_f32_i64             = ivf_pq_test<float, float, int64_t>;
using f32_f32_i64_flat_layout = ivf_pq_flat_layout_test<float, float, int64_t>;
using f32_f32_i64_filter      = ivf_pq_filter_test<float, float, int64_t>;

TEST_BUILD_HOST_INPUT_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_f32_i64)
TEST_BUILD_EXTEND_SEARCH(f32_f32_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_f32_i64)
TEST_BUILD_PRECOMPUTED(f32_f32_i64)
INSTANTIATE(f32_f32_i64,
            defaults() + small_dims() + big_dims_moderate_lut() + enum_variety_l2() +
              enum_variety_l2sqrt() + enum_variety_ip() + enum_variety_cosine());

TEST_FLAT_LAYOUT_CODES(f32_f32_i64_flat_layout)
INSTANTIATE(f32_f32_i64_flat_layout, flat_layout_tests());

TEST_BUILD_SEARCH(f32_f32_i64_filter)
INSTANTIATE(f32_f32_i64_filter,
            defaults() + small_dims() + big_dims_moderate_lut() + enum_variety_l2() +
              enum_variety_l2sqrt() + enum_variety_ip() + enum_variety_cosine());

template <typename MdspanT>
auto copy_to_host(raft::resources const& handle, MdspanT view)
{
  std::vector<std::remove_cv_t<typename MdspanT::element_type>> h(view.size());
  raft::update_host(
    h.data(), view.data_handle(), h.size(), raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  return h;
}

void check_ivf_pq_build_reproducible(codebook_gen codebook_kind)
{
  raft::resources handle;
  const int64_t n_rows = 100000;
  const int64_t dim    = 32;
  auto dataset         = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  raft::random::RngState rng(42ULL);
  raft::random::uniform(handle, rng, dataset.data_handle(), n_rows * dim, -1.0f, 1.0f);

  index_params params;
  params.n_lists       = 500;
  params.pq_dim        = 16;
  params.codebook_kind = codebook_kind;
  std::vector<std::vector<float>> centers, pq_centers;
  for (int run = 0; run < 2; run++) {
    const auto index = build(handle, params, raft::make_const_mdspan(dataset.view()));
    centers.push_back(copy_to_host(handle, index.centers()));
    pq_centers.push_back(copy_to_host(handle, index.pq_centers()));
  }
  ASSERT_EQ(0, std::memcmp(centers[0].data(), centers[1].data(), centers[0].size() * sizeof(float)))
    << "IVF-PQ cluster centers differ between identical builds";
  ASSERT_EQ(
    0,
    std::memcmp(pq_centers[0].data(), pq_centers[1].data(), pq_centers[0].size() * sizeof(float)))
    << "IVF-PQ codebooks differ between identical builds";
}

TEST(IvfPqBuild, ReproduciblePerSubspace)
{
  check_ivf_pq_build_reproducible(codebook_gen::PER_SUBSPACE);
}

TEST(IvfPqBuild, ReproduciblePerCluster)
{
  check_ivf_pq_build_reproducible(codebook_gen::PER_CLUSTER);
}

}  // namespace cuvs::neighbors::ivf_pq
