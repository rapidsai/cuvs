/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#pragma once

#include "../sample_filter.cuh"  // none_sample_filter
#include "ivf_pq_fp_8bit.cuh"    // cuvs::neighbors::ivf_pq::detail::fp_8bit

#include <cuvs/distance/distance.hpp>   // cuvs::distance::DistanceType
#include <cuvs/neighbors/ivf_pq.hpp>    // cuvs::neighbors::ivf_pq::codebook_gen
#include <raft/core/detail/macros.hpp>  // RAFT_WEAK_FUNCTION
#include <rmm/cuda_stream_view.hpp>     // rmm::cuda_stream_view

#include <cuda_fp16.h>  // __half

namespace cuvs::neighbors::ivf_pq::detail {

// is_local_topk_feasible is not inline here, because we would have to define it
// here as well. That would run the risk of the definitions here and in the
// -inl.cuh header diverging.
auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k, uint32_t n_probes, uint32_t n_queries)
  -> bool;

template <typename OutT,
          typename LutT,
          typename IvfSampleFilterT,
          uint32_t PqBits,
          int Capacity,
          bool PrecompBaseDiff,
          bool EnableSMemLut>
RAFT_KERNEL compute_similarity_kernel(uint32_t dim,
                                      uint32_t n_probes,
                                      uint32_t pq_dim,
                                      uint32_t n_queries,
                                      uint32_t queries_offset,
                                      distance::DistanceType metric,
                                      codebook_gen codebook_kind,
                                      uint32_t topk,
                                      uint32_t max_samples,
                                      const float* cluster_centers,
                                      const float* pq_centers,
                                      const uint8_t* const* pq_dataset,
                                      const uint32_t* cluster_labels,
                                      const uint32_t* _chunk_indices,
                                      const float* queries,
                                      const uint32_t* index_list,
                                      float* query_kths,
                                      IvfSampleFilterT sample_filter,
                                      LutT* lut_scores,
                                      OutT* _out_scores,
                                      uint32_t* _out_indices);

// The signature of the kernel defined by a minimal set of template parameters
template <typename OutT, typename LutT, typename IvfSampleFilterT>
using compute_similarity_kernel_t =
  decltype(&compute_similarity_kernel<OutT, LutT, IvfSampleFilterT, 8, 0, true, true>);

template <typename OutT, typename LutT, typename IvfSampleFilterT>
struct selected {
  compute_similarity_kernel_t<OutT, LutT, IvfSampleFilterT> kernel;
  dim3 grid_dim;
  dim3 block_dim;
  size_t smem_size;
  size_t device_lut_size;
};

template <typename OutT, typename LutT, typename IvfSampleFilterT>
void compute_similarity_run(selected<OutT, LutT, IvfSampleFilterT> s,
                            rmm::cuda_stream_view stream,
                            uint32_t dim,
                            uint32_t n_probes,
                            uint32_t pq_dim,
                            uint32_t n_queries,
                            uint32_t queries_offset,
                            distance::DistanceType metric,
                            codebook_gen codebook_kind,
                            uint32_t topk,
                            uint32_t max_samples,
                            const float* cluster_centers,
                            const float* pq_centers,
                            const uint8_t* const* pq_dataset,
                            const uint32_t* cluster_labels,
                            const uint32_t* _chunk_indices,
                            const float* queries,
                            const uint32_t* index_list,
                            float* query_kths,
                            IvfSampleFilterT sample_filter,
                            LutT* lut_scores,
                            OutT* _out_scores,
                            uint32_t* _out_indices);

/**
 * Use heuristics to choose an optimal instance of the search kernel.
 * It selects among a few kernel variants (with/out using shared mem for
 * lookup tables / precomputed distances) and tries to choose the block size
 * to maximize kernel occupancy.
 *
 * @param manage_local_topk
 *    whether use the fused calculate+select or just calculate the distances for each
 *    query and probed cluster.
 *
 * @param locality_hint
 *    beyond this limit do not consider increasing the number of active blocks per SM
 *    would improve locality anymore.
 */
template <typename OutT, typename LutT, typename IvfSampleFilterT>
auto compute_similarity_select(const cudaDeviceProp& dev_props,
                               bool manage_local_topk,
                               int locality_hint,
                               double preferred_shmem_carveout,
                               uint32_t pq_bits,
                               uint32_t pq_dim,
                               uint32_t precomp_data_count,
                               uint32_t n_queries,
                               uint32_t n_probes,
                               uint32_t topk) -> selected<OutT, LutT, IvfSampleFilterT>;

}  // namespace cuvs::neighbors::ivf_pq::detail

#define instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(                    \
  OutT, LutT, IvfSampleFilterT)                                                                \
  extern template auto                                                                         \
  cuvs::neighbors::ivf_pq::detail::compute_similarity_select<OutT, LutT, IvfSampleFilterT>(    \
    const cudaDeviceProp& dev_props,                                                           \
    bool manage_local_topk,                                                                    \
    int locality_hint,                                                                         \
    double preferred_shmem_carveout,                                                           \
    uint32_t pq_bits,                                                                          \
    uint32_t pq_dim,                                                                           \
    uint32_t precomp_data_count,                                                               \
    uint32_t n_queries,                                                                        \
    uint32_t n_probes,                                                                         \
    uint32_t topk) -> cuvs::neighbors::ivf_pq::detail::selected<OutT, LutT, IvfSampleFilterT>; \
                                                                                               \
  extern template void                                                                         \
  cuvs::neighbors::ivf_pq::detail::compute_similarity_run<OutT, LutT, IvfSampleFilterT>(       \
    cuvs::neighbors::ivf_pq::detail::selected<OutT, LutT, IvfSampleFilterT> s,                 \
    rmm::cuda_stream_view stream,                                                              \
    uint32_t dim,                                                                              \
    uint32_t n_probes,                                                                         \
    uint32_t pq_dim,                                                                           \
    uint32_t n_queries,                                                                        \
    uint32_t queries_offset,                                                                   \
    cuvs::distance::DistanceType metric,                                                       \
    cuvs::neighbors::ivf_pq::codebook_gen codebook_kind,                                       \
    uint32_t topk,                                                                             \
    uint32_t max_samples,                                                                      \
    const float* cluster_centers,                                                              \
    const float* pq_centers,                                                                   \
    const uint8_t* const* pq_dataset,                                                          \
    const uint32_t* cluster_labels,                                                            \
    const uint32_t* _chunk_indices,                                                            \
    const float* queries,                                                                      \
    const uint32_t* index_list,                                                                \
    float* query_kths,                                                                         \
    IvfSampleFilterT sample_filter,                                                            \
    LutT* lut_scores,                                                                          \
    OutT* _out_scores,                                                                         \
    uint32_t* _out_indices);

#define COMMA ,
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::none_sample_filter>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::none_sample_filter>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  half,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::none_sample_filter>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  half,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::none_sample_filter>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  float,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::none_sample_filter>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::none_sample_filter>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::none_sample_filter>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  half,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  half,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  float,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  cuvs::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  cuvs::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);

#undef COMMA

#undef instantiate_cuvs_neighbors_ivf_pq_detail_compute_similarity_select
