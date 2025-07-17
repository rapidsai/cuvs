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
#include "distance_base.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/random/rng.cuh>

namespace cuvs {
namespace distance {

template <typename DataType, typename OutputType = DataType>
void naive_impl(const DataType* x,
                                   const DataType* y,
                                   OutputType* dist,
                                   std::int64_t m,
                                   std::int64_t n,
                                   std::int64_t k,
                                   bool isRowMajor)
{
  // CPU implementation of bitwise hamming distance
  for (std::int64_t midx = 0; midx < m; ++midx) {
    for (std::int64_t nidx = 0; nidx < n; ++nidx) {
      OutputType acc = OutputType(0);
      
      for (std::int64_t i = 0; i < k; ++i) {
        std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
        std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
        
        uint8_t xv = static_cast<uint8_t>(x[xidx]);
        uint8_t yv = static_cast<uint8_t>(y[yidx]);
        
        acc += __builtin_popcount(xv ^ yv);
      }
      
      std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
      dist[outidx] = acc;
    }
  }
}

template <typename DataType, typename AccType, typename OutputType>
void naive_distance(raft::resources const& handle,
                                const DataType* x,
                                const DataType* y,
                                OutputType* dist,
                                std::int64_t m,
                                std::int64_t n,
                                std::int64_t k,
                                bool isRowMajor)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  
  std::vector<DataType> h_x(m * k);
  std::vector<DataType> h_y(n * k);
  std::vector<OutputType> h_dist(m * n);
  
  // Copy input data from device to host
  raft::copy(h_x.data(), x, m * k, stream);
  raft::copy(h_y.data(), y, n * k, stream);
  raft::resource::sync_stream(handle, stream);
  
  // Compute on CPU
  naive_impl(h_x.data(), h_y.data(), h_dist.data(), 
             m, n, k, isRowMajor);

  raft::copy(dist, h_dist.data(), m * n, stream);
}

struct BitwiseHammingInputs {
  double tolerance;
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
  bool isRowMajor;
  unsigned long long int seed;
};

template <typename OutputType>
class BitwiseHammingTest : public ::testing::TestWithParam<BitwiseHammingInputs> {
 public:
  BitwiseHammingTest()
    : params(::testing::TestWithParam<BitwiseHammingInputs>::GetParam()),
      handle{},
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      dist_ref(params.m * params.n, stream),
      dist_optimized(params.m * params.n, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    
    // Generate random uint8_t data
    raft::random::uniformInt(handle, r, x.data(), params.m * params.k, 
                            uint8_t(0), uint8_t(255));
    raft::random::uniformInt(handle, r, y.data(), params.n * params.k, 
                            uint8_t(0), uint8_t(255));
    naive_distance<uint8_t, uint32_t, OutputType>(
      handle, x.data(), y.data(), dist_ref.data(), 
      params.m, params.n, params.k, params.isRowMajor);
    
    // Compute optimized distance (will use uint32_t/uint64_t paths when possible)
    if (params.isRowMajor) {
      auto x_view = raft::make_device_matrix_view<const uint8_t, std::int64_t, raft::layout_c_contiguous>(
        x.data(), params.m, params.k);
      auto y_view = raft::make_device_matrix_view<const uint8_t, std::int64_t, raft::layout_c_contiguous>(
        y.data(), params.n, params.k);
      auto dist_view = raft::make_device_matrix_view<OutputType, std::int64_t, raft::layout_c_contiguous>(
        dist_optimized.data(), params.m, params.n);
      
      cuvs::distance::pairwise_distance(handle, x_view, y_view, dist_view, 
                                       cuvs::distance::DistanceType::BitwiseHamming);
    } else {
      auto x_view = raft::make_device_matrix_view<const uint8_t, std::int64_t, raft::layout_f_contiguous>(
        x.data(), params.m, params.k);
      auto y_view = raft::make_device_matrix_view<const uint8_t, std::int64_t, raft::layout_f_contiguous>(
        y.data(), params.n, params.k);
      auto dist_view = raft::make_device_matrix_view<OutputType, std::int64_t, raft::layout_f_contiguous>(
        dist_optimized.data(), params.m, params.n);
      
      cuvs::distance::pairwise_distance(handle, x_view, y_view, dist_view,
                                       cuvs::distance::DistanceType::BitwiseHamming);
    }
  }

 protected:
  BitwiseHammingInputs params;
  raft::resources handle;
  cudaStream_t stream;
  rmm::device_uvector<uint8_t> x;
  rmm::device_uvector<uint8_t> y;
  rmm::device_uvector<OutputType> dist_ref;
  rmm::device_uvector<OutputType> dist_optimized;
};

// Test cases focusing on different k values to test optimization paths
const std::vector<BitwiseHammingInputs> bitwiseHammingInputs = {
  // Test k divisible by 8 (should use uint64_t path)
  {0.0, 128, 128, 64, true, 1234ULL},
  {0.0, 256, 128, 128, true, 1234ULL},
  {0.0, 128, 256, 256, false, 1234ULL},
  {0.0, 512, 512, 512, true, 1234ULL},
  
  // Test k divisible by 4 but not 8 (should use uint32_t path)
  {0.0, 128, 128, 36, true, 1234ULL},
  {0.0, 256, 128, 100, true, 1234ULL},
  {0.0, 128, 256, 44, false, 1234ULL},
  {0.0, 256, 256, 124, true, 1234ULL},
  
  // Test k not divisible by 4 (should use uint8_t path)
  {0.0, 128, 128, 33, true, 1234ULL},
  {0.0, 256, 128, 67, true, 1234ULL},
  {0.0, 128, 256, 101, false, 1234ULL},
  {0.0, 200, 150, 39, true, 1234ULL},
  
  // Edge cases
  {0.0, 1, 1, 1, true, 1234ULL},
  {0.0, 17, 5, 3, true, 1234ULL},
  {0.0, 7, 23, 19, false, 1234ULL},
  
  // Large dimensions
  {0.0, 1024, 1024, 1024, true, 1234ULL},
  {0.0, 2048, 1024, 768, false, 1234ULL},
};

typedef BitwiseHammingTest<uint32_t> BitwiseHammingTestU32;

TEST_P(BitwiseHammingTestU32, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  
  // Compare optimized implementation against naive reference
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist_optimized.data(), m, n, 
    cuvs::CompareApprox<uint32_t>(params.tolerance), stream))
    << "Optimized BitwiseHamming distance does not match naive implementation"
    << " for k=" << params.k << " (divisible by 8: " << (params.k % 8 == 0)
    << ", divisible by 4: " << (params.k % 4 == 0) << ")";
}

INSTANTIATE_TEST_CASE_P(BitwiseHammingTests, BitwiseHammingTestU32, 
                        ::testing::ValuesIn(bitwiseHammingInputs));

}  // end namespace distance
}  // namespace cuvs 