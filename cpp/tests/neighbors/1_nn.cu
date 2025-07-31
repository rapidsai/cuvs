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
#include "1_nn_utils.cuh"

#include "../../src/distance/fused_distance_nn.cuh"
#include "../../src/distance/unfused_distance_nn.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>


namespace cuvs::neighbors {

  enum class ImplType {
    fused,
    unfused
  };




template <typename IdxT>
struct NNInputs {
  IdxT m;
  IdxT n;
  IdxT k;
  uint64_t rng_seed;
  double tol;
};

template <typename DataT, typename IdxT, ImplType impl>
class NNTest : public ::testing::TestWithParam<NNInputs<IdxT>> {
 public:
  using AccT = DataT;
  using OutT = raft::KeyValuePair<IdxT, AccT>;
  NNTest()
    : params_{::testing::TestWithParam<NNInputs<IdxT>>::GetParam()},
      m{params_.m},
      n{params_.n},
      k{params_.k},
      stream{raft::resource::get_cuda_stream(handle)},
      x{raft::make_device_matrix<DataT, IdxT>(handle, m, k)},
      y{raft::make_device_matrix<DataT, IdxT>(handle, n, k)},
      x_norm{raft::make_device_vector<AccT, IdxT>(handle, m)},
      y_norm{raft::make_device_vector<AccT, IdxT>(handle, n)},
      out{raft::make_device_vector<OutT, IdxT>(handle, m)},
      ref_out{raft::make_device_vector<OutT, IdxT>(handle, m)}
  {
  }

 protected:


  void SetUp() override
  {
    raft::random::RngState rng{params_.rng_seed};
    raft::random::uniform(
       handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(
       handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));

    // Pre-compute norms
    constexpr auto l2_row_norm = raft::linalg::rowNorm<raft::linalg::L2Norm, true, DataT, IdxT>;
    l2_row_norm(x_norm.data_handle(), x.data_handle(), k, m, stream, raft::identity_op());
    l2_row_norm(y_norm.data_handle(), y.data_handle(), k, n, stream, raft::identity_op());

    if constexpr (impl == ImplType::fused) {
      workspace_size = n * sizeof(IdxT);
    } else if constexpr (impl == ImplType::unfused) {
      workspace_size = m * n * sizeof(AccT);
    }

    // Reset buffers
    RAFT_CUDA_TRY(cudaMemsetAsync(out.data_handle(), 0, m * sizeof(OutT)));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  }

  void compute_1nn() {
    raft::device_vector<char, IdxT> workspace = raft::make_device_vector<char, IdxT>(handle, workspace_size);

    ref_l2nn_api<DataT, AccT, OutT, IdxT>(ref_out.data_handle(), x.data_handle(), y.data_handle(), m, n, k, stream);

    if constexpr (impl == ImplType::fused) {
      if constexpr (std::is_same_v<DataT, float>) {
      cuvs::distance::fusedDistanceNNMinReduce<DataT, OutT, IdxT>(out.data_handle(),
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
                                                                    stream);
      } else {
        static_assert(sizeof(DataT) == 0, "fusedDistanceNNMinReduce is not implemented for datatype other than float");
      }
    } else if constexpr (impl == ImplType::unfused) {
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
    }
  }

  void compare()
  {
    vector_compare(handle, ref_out.data_handle(), out.data_handle(), m, summary);
    //ASSERT_TRUE(summary.max_diff < params_.tol) << summary;
  }

 private:
  raft::resources handle;
  rmm::cuda_stream_view stream;
  NNInputs<IdxT> params_;
  ComparisonSummary summary;
  IdxT m;
  IdxT n;
  IdxT k;
  raft::device_matrix<DataT, IdxT> x;
  raft::device_matrix<DataT, IdxT> y;
  raft::device_vector<AccT, IdxT> x_norm;
  raft::device_vector<AccT, IdxT> y_norm;
  raft::device_vector<OutT, IdxT> out;
  raft::device_vector<OutT, IdxT> ref_out;
  size_t workspace_size;
};

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp32 = {
  {4096, 4096, 64, uint64_t(31415926), 0.1},
  {4096, 4096, 128, uint64_t(31415926), 0.1},
  };

// Test fused implementation with single-precision
typedef NNTest<float, int32_t, ImplType::fused> NNTest_fp32_fused;
TEST_P(NNTest_fp32_fused, test) {
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_fused, ::testing::ValuesIn(input_fp32<int>));

// Test unfused implementation with single-precision
typedef NNTest<float, int32_t, ImplType::unfused> NNTest_fp32_unfused;
TEST_P(NNTest_fp32_unfused, test) {
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_unfused, ::testing::ValuesIn(input_fp32<int>));

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp16 = {
  {4096, 4096, 64, uint64_t(31415926), 0.2},
  {4096, 4096, 128, uint64_t(31415926), 0.2},
  };

// Test unfused implementation with fp16
typedef NNTest<half, int32_t, ImplType::unfused> NNTest_fp16_unfused;
TEST_P(NNTest_fp16_unfused, test) {
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp16_unfused, ::testing::ValuesIn(input_fp16<int>));

} // namespace cuvs::neighbor
