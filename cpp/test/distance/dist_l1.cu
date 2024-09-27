/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
class DistanceUnexpL1
  : public DistanceTest<cuvs::distance::DistanceType::L1, DataType, OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceUnexpL1<float> DistanceUnexpL1F;
TEST_P(DistanceUnexpL1F, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceUnexpL1F, ::testing::ValuesIn(inputsf));

const std::vector<DistanceInputs<double>> inputsd = {
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001, 1024, 1024, 32, false, 1234ULL},
  {0.001, 1024, 32, 1024, false, 1234ULL},
  {0.001, 32, 1024, 1024, false, 1234ULL},
  {0.003, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceUnexpL1<double> DistanceUnexpL1D;
TEST_P(DistanceUnexpL1D, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceUnexpL1D, ::testing::ValuesIn(inputsd));

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
typedef DistanceUnexpL1<half, float> DistanceUnexpL1H;
TEST_P(DistanceUnexpL1H, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceUnexpL1H, ::testing::ValuesIn(inputsh));

class BigMatrixUnexpL1 : public BigMatrixDistanceTest<cuvs::distance::DistanceType::L1> {};
TEST_F(BigMatrixUnexpL1, Result) {}

}  // end namespace distance
}  // namespace cuvs
