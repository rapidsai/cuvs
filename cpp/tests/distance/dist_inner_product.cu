/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs::distance {

template <typename DataType, typename OutputType = DataType>
class DistanceInnerProduct
  : public DistanceTest<cuvs::distance::DistanceType::InnerProduct, DataType, OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {

  {0.001f, 10, 5, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceInnerProductF = DistanceInnerProduct<float>;
TEST_P(DistanceInnerProductF,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;

  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceInnerProductF,
  ::testing::ValuesIn(inputsf));  // NOLINT(modernize-use-trailing-return-type)

const std::vector<DistanceInputs<double>> inputsd = {

  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceInnerProductD = DistanceInnerProduct<double>;
TEST_P(DistanceInnerProductD,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;

  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceInnerProductD,
  ::testing::ValuesIn(inputsd));  // NOLINT(modernize-use-trailing-return-type)

const std::vector<DistanceInputs<half, float>> inputsh = {

  {0.001f, 10, 5, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceInnerProductH = DistanceInnerProduct<half, float>;
TEST_P(DistanceInnerProductH,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;

  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceInnerProductH,
  ::testing::ValuesIn(inputsh));  // NOLINT(modernize-use-trailing-return-type)

class BigMatrixInnerProduct
  : public BigMatrixDistanceTest<cuvs::distance::DistanceType::InnerProduct> {};
TEST_F(BigMatrixInnerProduct, Result) {}  // NOLINT(modernize-use-trailing-return-type)

}  // namespace cuvs::distance
