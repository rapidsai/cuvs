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

#include "../test_utils.cuh"
#include "../test_utils.h"
#include "./knn_utils.cuh"
#include <cuvs/distance/distance.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>
//#include <raft/matrix/init.cuh>
#include <cuda_fp16.h>

#include "../../src/distance/fused_distance_nn.cuh"
#include "../../src/distance/unfused_distance_nn.cuh"

template <typename Key, typename Value>
::std::ostream& operator<<(::std::ostream& os, const raft::KeyValuePair<Key, Value>& kv)
{
  os << "{ " << kv.key << ", " << kv.value << '}';
  return os;
}

namespace cuvs::neighbors {

template <typename DataT, typename IdxT>
struct NNInputs {
  IdxT m;
  IdxT n;
  IdxT k;
};

template <typename T>
__host__ __device__ T max_val() {
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MAX_NORMAL_FP16;
  } else {
    return cuda::std::numeric_limits<T>::max();
  }
}

template <typename T>
T min_val() {
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MIN_DENORM_FP16;
  } else {
    return cuda::std::numeric_limits<T>::min();
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__global__ void ref_l2nn_dev(OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K) {

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
  uint64_t max_diff_index;   // where does the maximim difference occur
  double max_a;
  double max_b;
  double acc_diff;          // sum of all the diffs
  uint64_t n;               // How many items are compared
  uint64_t n_misses;
  int mutex;                // Simple mutex lock for thread synchronization

  __device__ __host__
  void init() {
    max_diff = 0.0;
    max_diff_index = 0;
    max_a = 0.0;
    max_b = 0.0;
    acc_diff = 0.0;
    n = 0;
    n_misses = 0;
  }

  __device__ __host__
  void update(double diff, uint64_t index, double a_val, double b_val, bool missed) {
    if ( max_diff < diff ) {
      max_diff = diff;
      max_diff_index = index;
      max_a = a_val;
      max_b = b_val;
    }
    acc_diff += diff;
    n++;
    n_misses = missed ? n_misses + 1 : n_misses;
  }

  __device__ __host__
  void update(ComparisonSummary& op2) {
    if ( max_diff < op2.max_diff ) {
      max_diff = op2.max_diff;
      max_diff_index = op2.max_diff_index;
      max_a = op2.max_a;
      max_b = op2.max_b;
    }
    acc_diff += op2.acc_diff;
    n += op2.n;
    n_misses += op2.n_misses;
  }

  __device__ __host__
  void print() {
    if (max_diff > 0.0) {
      printf("Total compared %lu\n", n);
      printf("Total missed %lu\n", n_misses);
      printf("Average diff: %e\n", acc_diff / n);
      printf("max_diff: %e (%e - %e)\n", max_diff, max_a, max_b);
      printf("max_diff_index: %lu\n", max_diff_index);
    }
  }
};

template <typename OutT, typename IdxT>
__global__ void vector_compare_kernel(const OutT* a, const OutT* b, IdxT n,
                                      ComparisonSummary* global_summary) {
  // Shared memory for tracking failures within a block
  __shared__ ComparisonSummary block_summary;

  // Initialize shared memory variable
  if (threadIdx.x == 0) {
    block_summary.init();
  }
  __syncthreads();

  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (IdxT i = 0; i < n; i++) {
    tid = i;
    double diff, a_val, b_val;
    bool missed = false;
    if constexpr (std::is_fundamental_v<OutT> || std::is_same_v<OutT, half>) {
      diff = std::abs(double(a[tid]) - double(b[tid]));
      //printf("Expected = %f vs actual = %f\n", a[tid], b[tid]);
      a_val = double(a[tid]);
      b_val = double(b[tid]);

    } else {
      diff = std::abs(double(a[tid].value) - double(b[tid].value));
      //if (tid == 0) printf("Expected = %f vs actual = %f, %d, %d\n", a[tid].value, b[tid].value, a[tid].key, b[tid].key);
      a_val = double(a[tid].value);
      b_val = double(b[tid].value);

      missed = a[tid].key != b[tid].key;
    }
      // Acquire mutex lock using atomic compare-and-swap
    /*while (atomicCAS(&block_summary.mutex, 0, 1) != 0) {
      // Spin wait
    }*/

    // Critical section: update first_index if this is the earliest failure
    block_summary.update(diff, tid, a_val, b_val, missed);

    // Release mutex
    //atomicExch(&block_summary.mutex, 0);
  }

  __syncthreads();

  // First thread in the block can report the failure if needed
  if (threadIdx.x == 0) {
    // Acquire mutex lock using atomic compare-and-swap
    /*while (atomicCAS(&global_summary->mutex, 0, 1) != 0) {
      // Spin wait
    }*/

    // Critical section: update first_index if this is the earliest failure
    global_summary->update(block_summary);

    // Release mutex
    //atomicExch(&global_summary->mutex, 0);
  }
}

template <typename OutT, typename IdxT>
void vector_compare(ComparisonSummary* global_summary, const OutT* a, const OutT* b, const IdxT n, cudaStream_t stream = nullptr) {
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;


  //vector_compare_kernel<OutT, IdxT><<<grid_size, block_size, 0, stream>>>(a, b, n, global_summary);
  // Not thread safe right now, so launch only single thread
  vector_compare_kernel<OutT, IdxT><<<1, 1, 0, stream>>>(a, b, n, global_summary);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  //global_summary->print();

  //CHECK_CUDA(cudaFree(global_summary));
}
template <typename DataT, typename IdxT>
class NNTest : public ::testing::TestWithParam<NNInputs<DataT, IdxT>> {
 public:
  using AccT = DataT;
  using OutT = raft::KeyValuePair<IdxT, AccT>;
  NNTest()
    : params_(::testing::TestWithParam<NNInputs<DataT, IdxT>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle))
  {
  }

 protected:
  void test1NN()
  {
    /*raft::print_device_vector("Input array: ", input_.data(), rows_ * cols_, std::cout);
    std::cout << "K: " << k_ << std::endl;
    raft::print_device_vector("Labels array: ", search_labels_.data(), rows_, std::cout);

    auto index = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto search = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      (const T*)(search_data_.data()), rows_, cols_);
    auto indices =
      raft::make_device_matrix_view<IdxT, IdxT, raft::row_major>(indices_.data(), rows_, k_);
    auto distances =
      raft::make_device_matrix_view<DistT, IdxT, raft::row_major>(distances_.data(), rows_, k_);

    cuvs::neighbors::brute_force::index_params index_params;
    index_params.metric = cuvs::distance::DistanceType::L2Unexpanded;

    auto idx = cuvs::neighbors::brute_force::build(handle, index_params, index);
    cuvs::neighbors::brute_force::search_params search_params;
    cuvs::neighbors::brute_force::search(handle,
                                         search_params,
                                         idx,
                                         search,
                                         indices,
                                         distances,
                                         cuvs::neighbors::filtering::none_sample_filter{});

    build_actual_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      actual_labels_.data(), rows_, k_, search_labels_.data(), indices_.data());

    build_expected_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      expected_labels_.data(), rows_, k_, search_labels_.data());

    ASSERT_TRUE(devArrMatch(
      expected_labels_.data(), actual_labels_.data(), rows_ * k_, cuvs::Compare<int>(), stream));*/
  }

  void SetUp() override
  {
    IdxT m = params_.m;
    IdxT n = params_.n;
    IdxT k = params_.k;

    auto x     = raft::make_device_matrix<DataT, IdxT>(handle, m, k);
    auto y     = raft::make_device_matrix<DataT, IdxT>(handle, n, k);
    auto x_norm = raft::make_device_vector<AccT, IdxT>(handle, m);
    auto y_norm = raft::make_device_vector<AccT, IdxT>(handle, n);
    auto out   = raft::make_device_vector<OutT, IdxT>(handle, m);
    auto out_ref = raft::make_device_vector<OutT, IdxT>(handle, m);


    raft::random::RngState rng{1234};
    raft::random::uniform(
       handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(
       handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));


    // Pre-compute norms
    raft::linalg::rowNorm<raft::linalg::L2Norm,true,DataT,IdxT>(x_norm.data_handle(),
                          x.data_handle(),
                          k,
                          m,
                          stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm,true,DataT,IdxT>(y_norm.data_handle(),
                          y.data_handle(),
                          k,
                          n,
                          stream);

    size_t workspace_size = m * n * sizeof(AccT) > n * sizeof(IdxT) ? m * n * sizeof(AccT) : n * sizeof(IdxT);
    raft::device_vector<char, IdxT> workspace = raft::make_device_vector<char, IdxT>(handle, workspace_size);

    ref_l2nn_api<DataT, AccT, OutT, IdxT>(out_ref.data_handle(), x.data_handle(), y.data_handle(), m, n, k, stream);

    RAFT_CUDA_TRY(cudaMemsetAsync(workspace.data_handle(), 0, workspace_size, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(out.data_handle(), 0, m * sizeof(OutT)));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    /*cuvs::distance::fusedDistanceNNMinReduce<DataT, OutT, IdxT>(out.data_handle(),
                                                                x.data_handle(),
                                                                    y.data_handle(),
                                                                    x_norm.data_handle(),
                                                                    y_norm.data_handle(),
                                                                    static_cast<IdxT>(m),
                                                                    static_cast<IdxT>(n),
                                                                    static_cast<IdxT>(k),
                                                                    (void*)workspace.data_handle(),
                                                                    false,
                                                                    true,
                                                                    true,
                                                                    cuvs::distance::DistanceType::L2Expanded,
                                                                    0.0,
                                                                    stream);*/
    cuvs::distance::unfused_distance_nn<DataT, AccT, OutT, IdxT>(handle,
                 out.data_handle(),
                 x.data_handle(),
                 y.data_handle(),
                 m,
                 n,
                 k,
                 x_norm.data_handle(),
                 y_norm.data_handle(),
                 (AccT*)workspace.data_handle(),
                 false,
                 stream);
   

    ComparisonSummary* global_summary;
    RAFT_CUDA_TRY(cudaMallocManaged(&global_summary, sizeof(ComparisonSummary)));
    global_summary->init();

    vector_compare(global_summary, out_ref.data_handle(), out.data_handle(), m, stream);
    global_summary->print();
     /*
    actual_labels_.resize(rows_ * k_, stream);
    expected_labels_.resize(rows_ * k_, stream);
    input_.resize(rows_ * cols_, stream);
    search_data_.resize(rows_ * cols_, stream);
    indices_.resize(rows_ * k_, stream);
    distances_.resize(rows_ * k_, stream);
    search_labels_.resize(rows_, stream);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(actual_labels_.data(), 0, actual_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(expected_labels_.data(), 0, expected_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(input_.data(), 0, input_.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(search_data_.data(), 0, search_data_.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(indices_.data(), 0, indices_.size() * sizeof(IdxT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(distances_.data(), 0, distances_.size() * sizeof(DistT), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_labels_.data(), 0, search_labels_.size() * sizeof(int), stream));

    std::vector<T> row_major_input;
    for (std::size_t i = 0; i < params_.input.size(); ++i) {
      for (std::size_t j = 0; j < params_.input[i].size(); ++j) {
        row_major_input.push_back(params_.input[i][j]);
      }
    }
    rmm::device_buffer input_d =
      rmm::device_buffer(row_major_input.data(), row_major_input.size() * sizeof(T), stream);
    T* input_ptr = static_cast<T*>(input_d.data());

    rmm::device_buffer labels_d =
      rmm::device_buffer(params_.labels.data(), params_.labels.size() * sizeof(int), stream);
    int* labels_ptr = static_cast<int*>(labels_d.data());

    raft::copy(input_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_data_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_labels_.data(), labels_ptr, rows_, stream);
    raft::resource::sync_stream(handle, stream);*/
  }

 private:
  raft::resources handle;
  rmm::cuda_stream_view stream;

  NNInputs<DataT, IdxT> params_;
  /*int rows_;
  int cols_;
  rmm::device_uvector<T> input_;
  rmm::device_uvector<T> search_data_;
  rmm::device_uvector<IdxT> indices_;
  rmm::device_uvector<DistT> distances_;
  int k_;

  rmm::device_uvector<int> search_labels_;
  rmm::device_uvector<int> actual_labels_;
  rmm::device_uvector<int> expected_labels_;*/
};

template <typename DataT, typename IdxT>
const std::vector<NNInputs<DataT, IdxT>> inputs = {
  {4096, 4096, 128}
  };


typedef NNTest<float, int32_t> NNTest_float_int32_t;
TEST_P(NNTest_float_int32_t, Fused) { this->test1NN(); }

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_float_int32_t, ::testing::ValuesIn(inputs<float, int>));

}
