/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs {  // NOLINT(modernize-concat-nested-namespaces)
namespace distance {

template <typename DataType, typename OutputType = DataType>
class DistanceLinf : public DistanceTest<cuvs::distance::DistanceType::Linf,
                                         DataType,
                                         OutputType> {  // NOLINT(readability-identifier-naming)
};

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
typedef DistanceLinf<float>
  DistanceLinfF;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(DistanceLinfF,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceLinfF,
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
typedef DistanceLinf<double>
  DistanceLinfD;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(DistanceLinfD,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceLinfD,
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
typedef DistanceLinf<half, float>
  DistanceLinfH;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(DistanceLinfH,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceLinfH,
  ::testing::ValuesIn(
    inputsh));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

class BigMatrixLinf : public BigMatrixDistanceTest<cuvs::distance::DistanceType::Linf> {
};  // NOLINT(readability-identifier-naming)
TEST_F(BigMatrixLinf, Result) {
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

}  // end namespace distance
}  // namespace cuvs
