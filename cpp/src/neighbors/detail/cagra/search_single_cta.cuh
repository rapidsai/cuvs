/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "bitonic.hpp"
#include "compute_distance-ext.cuh"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "search_single_cta_kernel.cuh"
#include "topk_by_radix.cuh"
#include "topk_for_cagra/topk.h"  // TODO replace with raft topk
#include "utils.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace cuvs::neighbors::cagra::detail {
namespace single_cta_search {

template <typename DataT, typename IndexT, typename DistanceT, typename SAMPLE_FILTER_T>
struct search : search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T> {
  using base_type  = search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T>;
  using DATA_T     = typename base_type::DATA_T;
  using INDEX_T    = typename base_type::INDEX_T;
  using DISTANCE_T = typename base_type::DISTANCE_T;

  using base_type::algo;
  using base_type::hashmap_max_fill_rate;
  using base_type::hashmap_min_bitlen;
  using base_type::hashmap_mode;
  using base_type::itopk_size;
  using base_type::max_iterations;
  using base_type::max_queries;
  using base_type::min_iterations;
  using base_type::num_random_samplings;
  using base_type::rand_xor_mask;
  using base_type::search_width;
  using base_type::team_size;
  using base_type::thread_block_size;

  using base_type::dim;
  using base_type::graph_degree;
  using base_type::topk;

  using base_type::hash_bitlen;

  using base_type::dataset_size;
  using base_type::hashmap_size;
  using base_type::result_buffer_size;
  using base_type::small_hash_bitlen;
  using base_type::small_hash_reset_interval;

  using base_type::smem_size;

  using base_type::dataset_desc;
  using base_type::dev_seed;
  using base_type::hashmap;
  using base_type::num_executed_iterations;
  using base_type::num_seeds;

  uint32_t num_itopk_candidates;

  search(raft::resources const& res,
         search_params params,
         const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
         int64_t dim,
         int64_t graph_degree,
         uint32_t topk)
    : base_type(res, params, dataset_desc, dim, graph_degree, topk)
  {
    set_params(res);
  }

  ~search() {}

  inline void set_params(raft::resources const& res)
  {
    num_itopk_candidates = search_width * graph_degree;
    result_buffer_size   = itopk_size + num_itopk_candidates;

    typedef raft::Pow2<32> AlignBytes;
    unsigned result_buffer_size_32 = AlignBytes::roundUp(result_buffer_size);

    constexpr unsigned max_itopk = 512;
    RAFT_EXPECTS(itopk_size <= max_itopk, "itopk_size cannot be larger than %u", max_itopk);

    RAFT_LOG_DEBUG("# num_itopk_candidates: %u", num_itopk_candidates);
    RAFT_LOG_DEBUG("# num_itopk: %lu", itopk_size);
    //
    // Determine the thread block size
    //
    constexpr unsigned min_block_size       = 64;  // 32 or 64
    constexpr unsigned min_block_size_radix = 256;
    constexpr unsigned max_block_size       = 1024;
    //
    const std::uint32_t topk_ws_size = 3;
    const std::uint32_t base_smem_size =
      dataset_desc.smem_ws_size_in_bytes +
      (sizeof(INDEX_T) + sizeof(DISTANCE_T)) * result_buffer_size_32 +
      sizeof(INDEX_T) * hashmap::get_size(small_hash_bitlen) + sizeof(INDEX_T) * search_width +
      sizeof(std::uint32_t) * topk_ws_size + sizeof(std::uint32_t);
    smem_size = base_smem_size;
    if (num_itopk_candidates > 256) {
      // Tentatively calculate the required share memory size when radix
      // sort based topk is used, assuming the block size is the maximum.
      if (itopk_size <= 256) {
        smem_size += topk_by_radix_sort<256, INDEX_T>::smem_size * sizeof(std::uint32_t);
      } else {
        smem_size += topk_by_radix_sort<512, INDEX_T>::smem_size * sizeof(std::uint32_t);
      }
    }

    uint32_t block_size = thread_block_size;
    if (block_size == 0) {
      block_size = min_block_size;

      if (num_itopk_candidates > 256) {
        // radix-based topk is used.
        block_size = min_block_size_radix;

        // Internal topk values per thread must be equlal to or less than 4
        // when radix-sort block_topk is used.
        while ((block_size < max_block_size) && (max_itopk / block_size > 4)) {
          block_size *= 2;
        }
      }

      // Increase block size according to shared memory requirements.
      // If block size is 32, upper limit of shared memory size per
      // thread block is set to 4096. This is GPU generation dependent.
      constexpr unsigned ulimit_smem_size_cta32 = 4096;
      while (smem_size > ulimit_smem_size_cta32 / 32 * block_size) {
        block_size *= 2;
      }

      // Increase block size to improve GPU occupancy when batch size
      // is small, that is, number of queries is low.
      cudaDeviceProp deviceProp = raft::resource::get_device_properties(res);
      RAFT_LOG_DEBUG("# multiProcessorCount: %d", deviceProp.multiProcessorCount);
      while ((block_size < max_block_size) &&
             (graph_degree * search_width * team_size >= block_size * 2) &&
             (max_queries <= (1024 / (block_size * 2)) * deviceProp.multiProcessorCount)) {
        block_size *= 2;
      }
    }
    RAFT_LOG_DEBUG("# thread_block_size: %u", block_size);
    RAFT_EXPECTS(block_size >= min_block_size,
                 "block_size cannot be smaller than min_block size, %u",
                 min_block_size);
    RAFT_EXPECTS(block_size <= max_block_size,
                 "block_size cannot be larger than max_block size %u",
                 max_block_size);
    thread_block_size = block_size;

    if (num_itopk_candidates <= 256) {
      RAFT_LOG_DEBUG("# bitonic-sort based topk routine is used");
    } else {
      RAFT_LOG_DEBUG("# radix-sort based topk routine is used");
      smem_size = base_smem_size;
      if (itopk_size <= 256) {
        constexpr unsigned MAX_ITOPK = 256;
        smem_size += topk_by_radix_sort<MAX_ITOPK, INDEX_T>::smem_size * sizeof(std::uint32_t);
      } else {
        constexpr unsigned MAX_ITOPK = 512;
        smem_size += topk_by_radix_sort<MAX_ITOPK, INDEX_T>::smem_size * sizeof(std::uint32_t);
      }
    }
    RAFT_LOG_DEBUG("# smem_size: %u", smem_size);
    hashmap_size = 0;
    if (small_hash_bitlen == 0) {
      hashmap_size = sizeof(INDEX_T) * max_queries * hashmap::get_size(hash_bitlen);
      hashmap.resize(hashmap_size, raft::resource::get_cuda_stream(res));
    }
    RAFT_LOG_DEBUG("# hashmap_size: %lu", hashmap_size);
  }

  void operator()(raft::resources const& res,
                  raft::device_matrix_view<const INDEX_T, int64_t, raft::row_major> graph,
                  INDEX_T* const result_indices_ptr,       // [num_queries, topk]
                  DISTANCE_T* const result_distances_ptr,  // [num_queries, topk]
                  const DATA_T* const queries_ptr,         // [num_queries, dataset_dim]
                  const std::uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,                   // [num_queries, num_seeds]
                  std::uint32_t* const num_executed_iterations,  // [num_queries]
                  uint32_t topk,
                  SAMPLE_FILTER_T sample_filter)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(res);
    select_and_run(dataset_desc.dev_ptr(),
                   graph,
                   result_indices_ptr,
                   result_distances_ptr,
                   queries_ptr,
                   num_queries,
                   dev_seed_ptr,
                   num_executed_iterations,
                   *this,
                   topk,
                   num_itopk_candidates,
                   static_cast<uint32_t>(thread_block_size),
                   smem_size,
                   hash_bitlen,
                   hashmap.data(),
                   small_hash_bitlen,
                   small_hash_reset_interval,
                   num_seeds,
                   sample_filter,
                   stream);
  }
};

}  // namespace single_cta_search
}  // namespace cuvs::neighbors::cagra::detail
