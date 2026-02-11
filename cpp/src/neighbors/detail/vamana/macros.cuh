/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::vamana::detail {

/* Macros to compute the shared memory requirements for CUB primitives used by search and prune */
#define COMPUTE_SMEM_SIZE(degree, visited_size, DEG, CANDS)                                      \
  if (degree == DEG && visited_size <= CANDS && visited_size > CANDS / 2) {                      \
    sort_smem_size = static_cast<int>(                                                           \
      sizeof(typename cub::BlockMergeSort<dist_pair<IdxT, AccT>, 32, CANDS / 32>::TempStorage)); \
  }

// Current supported sizes for degree and visited_size. Note that visited_size must be > degree
#define SELECT_SORT_SMEM_SIZE(degree, visited_size)  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 64);   \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 128);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 256);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 512);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 128);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 256);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 512);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 128, 256); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 128, 512); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 256, 512); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 256, 1024);

/* Macros to call the CUB BlockSort primitives for supported sizes for GREEDY SEARCH */
#define SEARCH_CALL_SORT(topk, CANDS)                                              \
  if (topk <= CANDS && topk > CANDS / 2) {                                         \
    using BlockSortT = cub::BlockMergeSort<dist_pair<IdxT, AccT>, 32, CANDS / 32>; \
    auto& sort_mem   = reinterpret_cast<typename BlockSortT::TempStorage&>(smem);  \
    sort_visited<AccT, IdxT, CANDS>(&query_list[i], &sort_mem);                    \
  }

// SEARCH only relies on visited_size (not degree) for shared memory.
#define SEARCH_SELECT_SORT(topk) \
  SEARCH_CALL_SORT(topk, 64);    \
  SEARCH_CALL_SORT(topk, 128);   \
  SEARCH_CALL_SORT(topk, 256);   \
  SEARCH_CALL_SORT(topk, 512);   \
  SEARCH_CALL_SORT(topk, 1024);

}  // namespace cuvs::neighbors::vamana::detail
