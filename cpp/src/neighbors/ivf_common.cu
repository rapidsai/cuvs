/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/pow2_utils.cuh>

#include <cub/cub.cuh>

namespace cuvs::neighbors::ivf::detail {

/**
 * For each query, we calculate a cumulative sum of the cluster sizes that we probe, and return that
 * in chunk_indices. Essentially this is a segmented inclusive scan of the cluster sizes. The total
 * number of samples per query (sum of the cluster sizes that we probe) is returned in n_samples.
 */
template <int BlockDim>
__launch_bounds__(BlockDim) __global__
  void calc_chunk_indices_kernel(uint32_t n_probes,
                                 const uint32_t* cluster_sizes,      // [n_clusters]
                                 const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                                 uint32_t* chunk_indices,            // [n_queries, n_probes]
                                 uint32_t* n_samples                 // [n_queries]
  )
{
  using block_scan = cub::BlockScan<uint32_t, BlockDim>;
  __shared__ typename block_scan::TempStorage shm;

  // locate the query data
  clusters_to_probe += n_probes * blockIdx.x;
  chunk_indices += n_probes * blockIdx.x;

  // block scan
  const uint32_t n_probes_aligned = raft::Pow2<BlockDim>::roundUp(n_probes);
  uint32_t total                  = 0;
  for (uint32_t probe_ix = threadIdx.x; probe_ix < n_probes_aligned; probe_ix += BlockDim) {
    auto label = probe_ix < n_probes ? clusters_to_probe[probe_ix] : 0u;
    auto chunk = probe_ix < n_probes ? cluster_sizes[label] : 0u;
    if (threadIdx.x == 0) { chunk += total; }
    block_scan(shm).InclusiveSum(chunk, chunk, total);
    __syncthreads();
    if (probe_ix < n_probes) { chunk_indices[probe_ix] = chunk; }
  }
  // save the total size
  if (threadIdx.x == 0) { n_samples[blockIdx.x] = total; }
}

template __launch_bounds__(32) __global__ void calc_chunk_indices_kernel<32>(
  uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*);
template __launch_bounds__(64) __global__ void calc_chunk_indices_kernel<64>(
  uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*);
template __launch_bounds__(128) __global__ void calc_chunk_indices_kernel<128>(
  uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*);
template __launch_bounds__(256) __global__ void calc_chunk_indices_kernel<256>(
  uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*);
template __launch_bounds__(512) __global__ void calc_chunk_indices_kernel<512>(
  uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*);
template __launch_bounds__(1024) __global__ void calc_chunk_indices_kernel<1024>(
  uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t*);

}  // namespace cuvs::neighbors::ivf::detail
