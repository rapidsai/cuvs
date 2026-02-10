/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs {  // NOLINT(modernize-concat-nested-namespaces)
namespace distance {

template <typename DataType, typename OutputType = DataType>
class DistanceHamming  // NOLINT(readability-identifier-naming)
  : public DistanceTest<cuvs::distance::DistanceType::HammingUnexpanded, DataType, OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  // NOLINT(readability-identifier-naming)
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceHammingF = DistanceHamming<float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceHammingF,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceHammingF,
  ::testing::ValuesIn(
    inputsf));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

const std::vector<DistanceInputs<double>> inputsd = {
  // NOLINT(readability-identifier-naming)
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001, 1024, 1024, 32, false, 1234ULL},
  {0.001, 1024, 32, 1024, false, 1234ULL},
  {0.001, 32, 1024, 1024, false, 1234ULL},
  {0.003, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceHammingD = DistanceHamming<double>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceHammingD,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceHammingD,
  ::testing::ValuesIn(
    inputsd));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

const std::vector<DistanceInputs<half, float>> inputsh = {
  // NOLINT(readability-identifier-naming)
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceHammingH = DistanceHamming<half, float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceHammingH,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceHammingH,
  ::testing::ValuesIn(
    inputsh));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

class BigMatrixHamming  // NOLINT(readability-identifier-naming)
  : public BigMatrixDistanceTest<cuvs::distance::DistanceType::HammingUnexpanded> {};
TEST_F(BigMatrixHamming, Result) {
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
}  // end namespace distance
}  // namespace cuvs
