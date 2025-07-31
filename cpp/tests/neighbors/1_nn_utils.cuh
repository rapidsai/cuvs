
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuda_fp16.h>
#include <raft/util/cuda_rt_essentials.hpp>
#include <cuda/std/limits>
#include <raft/core/kvp.hpp>
#include <raft/core/host_mdarray.hpp>

namespace cuvs::neighbors {

template <typename T>
_RAFT_HOST_DEVICE T max_val() {
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MAX_NORMAL_FP16;
  } else {
    return cuda::std::numeric_limits<T>::max();
  }
}

template <typename T>
_RAFT_HOST_DEVICE T min_val() {
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MIN_DENORM_FP16;
  } else {
    return cuda::std::numeric_limits<T>::min();
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
RAFT_KERNEL ref_l2nn_dev(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K) {

  IdxT tid = threadIdx.x + blockIdx.x * size_t(blockDim.x);
  IdxT n_warps = (size_t(blockDim.x) * gridDim.x) / 32;

  IdxT warp_id = tid / 32;
  IdxT warp_lane = threadIdx.x % 32;
  const int warp_size = 32;

  for (IdxT m = warp_id; m < M; m+=n_warps) {
    __shared__ AccT dist[4];

    IdxT min_index = N + 1;
    AccT min_dist = max_val<AccT>();

    for (IdxT n = 0; n < N; n++) {
      if (warp_lane == 0) {
        dist[warp_id % 4] = AccT(0.0);
      }
      AccT th_dist = AccT(0.0);
      for (IdxT k = warp_lane; k < K; k+=warp_size) {
        AccT diff = AccT(A[m * K + k]) - AccT(B[n * K + k]);
        th_dist += (diff * diff);
      }
      __syncwarp();
      atomicAdd(&dist[warp_id % 4], th_dist);
      __syncwarp();

      if (warp_lane == 0 && dist[warp_id % 4] < min_dist) {
        min_dist = dist[warp_id % 4];
        min_index = n;
      }
    }
    if constexpr (std::is_fundamental<OutT>::value) {
      if (warp_lane == 0) {
        static_assert(std::is_same<OutT, AccT>::value, "OutT and AccT are not same type");
        out[m] = AccT(min_dist);
      }
    } else {
      // output is a raft::KeyValuePair
      if (warp_lane == 0) {
        static_assert(std::is_same<OutT, raft::KeyValuePair<IdxT, AccT>>::value, "OutT is not raft::KeyValuePair<> type");
        out[m].key = IdxT(min_index);
        out[m].value = AccT(min_dist);
      }
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void ref_l2nn_api(OutT* out, const DataT* A, const DataT* B, IdxT m, IdxT n, IdxT k, cudaStream_t stream) {

  //constexpr int block_dim = 128;
  //static_assert(block_dim % 32 == 0, "blockdim must be divisible by 32");
  //constexpr int warps_per_block = block_dim / 32;
  //int num_blocks = m ;
  ref_l2nn_dev<DataT, AccT, OutT, IdxT><<<m/4, 128, 0, stream>>>(out, A, B, m, n, k);
  return;
}

// Structure to track comparison failures
class ComparisonSummary {
public:
  double max_diff;          // Maximum difference found
  uint64_t max_diff_index;  // where does the maximum difference occur
  double max_diff_a;        // What was the `a` value at max difference
  double max_diff_b;        // What was the `b` value at max difference
  double acc_diff;          // sum of all the diffs
  uint64_t n;               // How many items are compared
  uint64_t n_misses;        // How many were wrong
  int mutex;                // Simple mutex lock for thread synchronization

  void init() {
    max_diff = 0.0;
    max_diff_index = 0;
    max_diff_a = 0.0;
    max_diff_b = 0.0;
    acc_diff = 0.0;
    n = 0;
    n_misses = 0;
  }

  void update(double diff, uint64_t index, double a_val, double b_val, bool missed) {
    if ( max_diff < diff ) {
      max_diff = diff;
      max_diff_index = index;
      max_diff_a = a_val;
      max_diff_b = b_val;
    }
    acc_diff += diff;
    n++;
    n_misses = missed ? n_misses + 1 : n_misses;
  }

  friend std::ostream& operator<<(std::ostream& os, const ComparisonSummary& summary) {
    if (summary.max_diff > 0.0) {
      os << "Total compared " << summary.n << std::endl;
      os << "Total missed " << summary.n_misses << std::endl;
      os << "Average diff: " << summary.acc_diff / summary.n << std::endl;
      os << "max_diff: " << summary.max_diff << " (" << summary.max_diff_a << " - " << summary.max_diff_b << ")" << std::endl;
      os << "max_diff_index: " << summary.max_diff_index << std::endl;
    }
    return os;
  }
};

template <typename OutT, typename IdxT>
void vector_compare(raft::resources const& handle, const OutT* a, const OutT* b, IdxT n, ComparisonSummary& summary) {

  auto a_h = raft::make_host_vector<OutT, IdxT>(n);
  auto b_h = raft::make_host_vector<OutT, IdxT>(n);

  raft::copy(a_h.data_handle(),
             a,
             n,
             raft::resource::get_cuda_stream(handle));

  raft::copy(b_h.data_handle(),
             b,
             n,
             raft::resource::get_cuda_stream(handle));

  summary.init();

  for (IdxT i = 0; i < n; i++) {
    double diff, a_val, b_val;
    bool missed = false;
    if constexpr (std::is_fundamental_v<OutT> || std::is_same_v<OutT, half>) {
      // OutT is float, half or an integer type
      a_val = double(a_h(i));
      b_val = double(b_h(i));
    } else {
      // OutT is a raft::KeyValuePair
      a_val = double(a_h(i).value);
      b_val = double(b_h(i).value);

      missed = a_h(i).key != b_h(i).key;
    }

    diff = std::abs(a_val - b_val);
    summary.update(diff, i, a_val, b_val, missed);
  }
}

}
