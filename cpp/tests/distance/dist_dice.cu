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

namespace cuvs {
namespace distance {

template <typename DataType, typename OutputType = DataType>
class DistanceExpDice
  : public DistanceTest<cuvs::distance::DistanceType::DiceExpanded, DataType, OutputType> {};

template <typename DataType, typename OutputType = DataType>
class DistanceExpDiceXequalY
  : public DistanceTestSameBuffer<cuvs::distance::DistanceType::DiceExpanded,
                                  DataType,
                                  OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsXeqYf = {
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsh = {
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};

typedef DistanceExpDice<float> DistanceExpDiceF;
TEST_P(DistanceExpDiceF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApproxNaN<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceF, ::testing::ValuesIn(inputsf));

typedef DistanceExpDice<half, float> DistanceExpDiceH;
TEST_P(DistanceExpDiceH, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApproxNaN<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceH, ::testing::ValuesIn(inputsh));

typedef DistanceExpDiceXequalY<float> DistanceExpDiceXequalYF;
TEST_P(DistanceExpDiceXequalYF, Result)
{
  int m = params.m;
  int n = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                n,
                                cuvs::CompareApproxNaN<float>(params.tolerance),
                                stream));
  n = params.isRowMajor ? m : m / 2;
  m = params.isRowMajor ? m / 2 : m;

  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m,
                                n,
                                cuvs::CompareApproxNaN<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceXequalYF, ::testing::ValuesIn(inputsXeqYf));

typedef DistanceExpDiceXequalY<half, float> DistanceExpDiceXequalYH;
TEST_P(DistanceExpDiceXequalYH, Result)
{
  int m = params.m;
  int n = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                n,
                                cuvs::CompareApproxNaN<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceXequalYH, ::testing::ValuesIn(inputsh));

const std::vector<DistanceInputs<float>> inputsd = {
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceExpDice<float> DistanceExpDiceD;
TEST_P(DistanceExpDiceD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApproxNaN<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceD, ::testing::ValuesIn(inputsd));

class BigMatrixDice : public BigMatrixDistanceTest<cuvs::distance::DistanceType::DiceExpanded> {};
TEST_F(BigMatrixDice, Result) {}

// Simple test to verify half precision works
TEST(DistanceExpDiceH, SimpleHalfTest)
{
  raft::resources handle;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  const int m = 32;
  const int n = 32;
  const int k = 16;

  rmm::device_uvector<half> x(m * k, stream);
  rmm::device_uvector<half> y(n * k, stream);
  rmm::device_uvector<float> dist(m * n, stream);

  // Initialize with simple values
  raft::random::RngState r(1234ULL);
  uniform(handle, r, x.data(), m * k, half(0.0), half(1.0));
  uniform(handle, r, y.data(), n * k, half(0.0), half(1.0));

  // Create views
  auto x_v =
    raft::make_device_matrix_view<half, std::int64_t, raft::layout_c_contiguous>(x.data(), m, k);
  auto y_v =
    raft::make_device_matrix_view<half, std::int64_t, raft::layout_c_contiguous>(y.data(), n, k);
  auto dist_v = raft::make_device_matrix_view<float, std::int64_t, raft::layout_c_contiguous>(
    dist.data(), m, n);

  // Compute distance
  cuvs::distance::pairwise_distance(
    handle, x_v, y_v, dist_v, cuvs::distance::DistanceType::DiceExpanded, 2.0f);

  // Just verify it doesn't crash and produces reasonable values
  raft::resource::sync_stream(handle);

  // Basic sanity check - distances should be in [0, 1] range for Dice
  ASSERT_TRUE(devArrMatch(
    0.0f,
    dist.data(),
    m * n,
    [](float expected, float actual) { return actual >= -0.1f && actual <= 1.1f; },
    stream));
}

}  // end namespace distance
}  // end namespace cuvs
