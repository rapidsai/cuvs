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

#pragma once

namespace cuvs::neighbors::vamana::detail {

/* Macros to compute the shared memory requirements for CUB primitives used by search and prune */
#define COMPUTE_SMEM_SIZES(degree, visited_size, DEG, CANDS)                                     \
  if (degree == DEG && visited_size <= CANDS && visited_size > CANDS / 2) {                      \
    search_smem_sort_size = static_cast<int>(                                                    \
      sizeof(typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, CANDS / 32>::TempStorage));  \
                                                                                                 \
    prune_smem_sort_size = static_cast<int>(sizeof(                                              \
      typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (CANDS + DEG) / 32>::TempStorage)); \
  }

// Current supported sizes for degree and visited_size. Note that visited_size must be > degree
#define SELECT_SMEM_SIZES(degree, visited_size)       \
  COMPUTE_SMEM_SIZES(degree, visited_size, 32, 64);   \
  COMPUTE_SMEM_SIZES(degree, visited_size, 32, 128);  \
  COMPUTE_SMEM_SIZES(degree, visited_size, 32, 256);  \
  COMPUTE_SMEM_SIZES(degree, visited_size, 32, 512);  \
  COMPUTE_SMEM_SIZES(degree, visited_size, 64, 128);  \
  COMPUTE_SMEM_SIZES(degree, visited_size, 64, 256);  \
  COMPUTE_SMEM_SIZES(degree, visited_size, 64, 512);  \
  COMPUTE_SMEM_SIZES(degree, visited_size, 128, 256); \
  COMPUTE_SMEM_SIZES(degree, visited_size, 128, 512); \
  COMPUTE_SMEM_SIZES(degree, visited_size, 256, 512); \
  COMPUTE_SMEM_SIZES(degree, visited_size, 256, 1024);

/* Macros to call the CUB BlockSort primitives for supported sizes for ROBUST_PRUNE*/
#define PRUNE_CALL_SORT(degree, visited_list, DEG, CANDS)                                  \
  if (degree == DEG && visited_list <= CANDS && visited_list > CANDS / 2) {                \
    using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (DEG + CANDS) / 32>;  \
    auto& sort_mem   = reinterpret_cast<typename BlockSortT::TempStorage&>(smem);          \
    sort_edges_and_cands<accT, IdxT, DEG, CANDS>(new_nbh_list, &query_list[i], &sort_mem); \
  }

#define PRUNE_SELECT_SORT(degree, visited_list)    \
  PRUNE_CALL_SORT(degree, visited_size, 32, 64);   \
  PRUNE_CALL_SORT(degree, visited_size, 32, 128);  \
  PRUNE_CALL_SORT(degree, visited_size, 32, 256);  \
  PRUNE_CALL_SORT(degree, visited_size, 32, 512);  \
  PRUNE_CALL_SORT(degree, visited_size, 64, 128);  \
  PRUNE_CALL_SORT(degree, visited_size, 64, 256);  \
  PRUNE_CALL_SORT(degree, visited_size, 64, 512);  \
  PRUNE_CALL_SORT(degree, visited_size, 128, 256); \
  PRUNE_CALL_SORT(degree, visited_size, 128, 512); \
  PRUNE_CALL_SORT(degree, visited_size, 256, 512); \
  PRUNE_CALL_SORT(degree, visited_size, 256, 1024);

/* Macros to call the CUB BlockSort primitives for supported sizes for GREEDY SEARCH */
#define SEARCH_CALL_SORT(topk, CANDS)                                             \
  if (topk <= CANDS && topk > CANDS / 2) {                                        \
    using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, CANDS / 32>; \
    auto& sort_mem   = reinterpret_cast<typename BlockSortT::TempStorage&>(smem); \
    sort_visited<accT, IdxT, CANDS>(&query_list[i], &sort_mem);                   \
  }

// SEARCH only relies on visited_size (not degree) for shared memory.
#define SEARCH_SELECT_SORT(topk) \
  SEARCH_CALL_SORT(topk, 64);    \
  SEARCH_CALL_SORT(topk, 128);   \
  SEARCH_CALL_SORT(topk, 256);   \
  SEARCH_CALL_SORT(topk, 512);   \
  SEARCH_CALL_SORT(topk, 1024);

}  // namespace cuvs::neighbors::vamana::detail
