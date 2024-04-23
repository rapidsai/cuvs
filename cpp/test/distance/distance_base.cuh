/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cuvs/distance/distance_types.hpp>  // cuvs::distance::DistanceType
#include <cuvs/distance/pairwise_distance.hpp>
#include <raft/common/nvtx.hpp>         // raft::common::nvtx::range
#include <raft/core/device_mdspan.hpp>  //raft::make_device_matrix_view
#include <raft/core/operators.hpp>      // raft::sqrt
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>  // raft::resources
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>  // rmm::device_uvector

#include <gtest/gtest.h>

namespace cuvs {
namespace distance {

template <typename DataType>
RAFT_KERNEL naiveDistanceKernel(DataType* dist,
                                const DataType* x,
                                const DataType* y,
                                std::int64_t m,
                                std::int64_t n,
                                std::int64_t k,
                                cuvs::distance::DistanceType type,
                                bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto diff         = x[xidx] - y[yidx];
    acc += diff * diff;
  }
  if (type == cuvs::distance::DistanceType::L2SqrtExpanded ||
      type == cuvs::distance::DistanceType::L2SqrtUnexpanded)
    acc = raft::sqrt(acc);
  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType>
RAFT_KERNEL naiveL1_Linf_CanberraDistanceKernel(DataType* dist,
                                                const DataType* x,
                                                const DataType* y,
                                                std::int64_t m,
                                                std::int64_t n,
                                                std::int64_t k,
                                                cuvs::distance::DistanceType type,
                                                bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc = DataType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    auto diff         = (a > b) ? (a - b) : (b - a);
    if (type == cuvs::distance::DistanceType::Linf) {
      acc = raft::max(acc, diff);
    } else if (type == cuvs::distance::DistanceType::Canberra) {
      const auto add = raft::abs(a) + raft::abs(b);
      // deal with potential for 0 in denominator by
      // forcing 1/0 instead
      acc += ((add != 0) * diff / (add + (add == 0)));
    } else {
      acc += diff;
    }
  }

  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType>
RAFT_KERNEL naiveCosineDistanceKernel(DataType* dist,
                                      const DataType* x,
                                      const DataType* y,
                                      std::int64_t m,
                                      std::int64_t n,
                                      std::int64_t k,
                                      bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc_a  = DataType(0);
  DataType acc_b  = DataType(0);
  DataType acc_ab = DataType(0);

  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    acc_a += a * a;
    acc_b += b * b;
    acc_ab += a * b;
  }

  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;

  // Use 1.0 - (cosine similarity) to calc the distance
  dist[outidx] = (DataType)1.0 - acc_ab / (raft::sqrt(acc_a) * raft::sqrt(acc_b));
}

template <typename DataType>
RAFT_KERNEL naiveInnerProductKernel(DataType* dist,
                                    const DataType* x,
                                    const DataType* y,
                                    std::int64_t m,
                                    std::int64_t n,
                                    std::int64_t k,
                                    bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc_ab = DataType(0);

  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    acc_ab += a * b;
  }

  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc_ab;
}

template <typename DataType>
RAFT_KERNEL naiveHellingerDistanceKernel(DataType* dist,
                                         const DataType* x,
                                         const DataType* y,
                                         std::int64_t m,
                                         std::int64_t n,
                                         std::int64_t k,
                                         bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc_ab = DataType(0);

  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    acc_ab += raft::sqrt(a) * raft::sqrt(b);
  }

  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;

  // Adjust to replace NaN in sqrt with 0 if input to sqrt is negative
  acc_ab         = 1 - acc_ab;
  auto rectifier = (!signbit(acc_ab));
  dist[outidx]   = raft::sqrt(rectifier * acc_ab);
}

template <typename DataType>
RAFT_KERNEL naiveLpUnexpDistanceKernel(DataType* dist,
                                       const DataType* x,
                                       const DataType* y,
                                       std::int64_t m,
                                       std::int64_t n,
                                       std::int64_t k,
                                       bool isRowMajor,
                                       DataType p)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    auto diff         = raft::abs(a - b);
    acc += raft::pow(diff, p);
  }
  auto one_over_p     = 1 / p;
  acc                 = raft::pow(acc, one_over_p);
  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType>
RAFT_KERNEL naiveHammingDistanceKernel(DataType* dist,
                                       const DataType* x,
                                       const DataType* y,
                                       std::int64_t m,
                                       std::int64_t n,
                                       std::int64_t k,
                                       bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    acc += (a != b);
  }
  acc                 = acc / k;
  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType>
RAFT_KERNEL naiveJensenShannonDistanceKernel(DataType* dist,
                                             const DataType* x,
                                             const DataType* y,
                                             std::int64_t m,
                                             std::int64_t n,
                                             std::int64_t k,
                                             bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];

    DataType m  = 0.5f * (a + b);
    bool a_zero = a == 0;
    bool b_zero = b == 0;

    DataType p = (!a_zero * m) / (a_zero + a);
    DataType q = (!b_zero * m) / (b_zero + b);

    bool p_zero = p == 0;
    bool q_zero = q == 0;

    acc += (-a * (!p_zero * log(p + p_zero))) + (-b * (!q_zero * log(q + q_zero)));
  }
  acc                 = raft::sqrt(0.5f * acc);
  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType, typename OutType>
RAFT_KERNEL naiveRussellRaoDistanceKernel(OutType* dist,
                                          const DataType* x,
                                          const DataType* y,
                                          std::int64_t m,
                                          std::int64_t n,
                                          std::int64_t k,
                                          bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  OutType acc = OutType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    acc += (a * b);
  }
  acc                 = (k - acc) / k;
  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType, typename OutType>
RAFT_KERNEL naiveKLDivergenceDistanceKernel(OutType* dist,
                                            const DataType* x,
                                            const DataType* y,
                                            std::int64_t m,
                                            std::int64_t n,
                                            std::int64_t k,
                                            bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  OutType acc = OutType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    bool b_zero       = (b == 0);
    bool a_zero       = (a == 0);
    acc += a * (log(a + a_zero) - log(b + b_zero));
  }
  acc                 = 0.5f * acc;
  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType, typename OutType>
RAFT_KERNEL naiveCorrelationDistanceKernel(OutType* dist,
                                           const DataType* x,
                                           const DataType* y,
                                           std::int64_t m,
                                           std::int64_t n,
                                           std::int64_t k,
                                           bool isRowMajor)
{
  std::int64_t midx = threadIdx.x + blockIdx.x * blockDim.x;
  std::int64_t nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  OutType acc    = OutType(0);
  auto a_norm    = DataType(0);
  auto b_norm    = DataType(0);
  auto a_sq_norm = DataType(0);
  auto b_sq_norm = DataType(0);
  for (std::int64_t i = 0; i < k; ++i) {
    std::int64_t xidx = isRowMajor ? i + midx * k : i * m + midx;
    std::int64_t yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    a_norm += a;
    b_norm += b;
    a_sq_norm += (a * a);
    b_sq_norm += (b * b);
    acc += (a * b);
  }

  auto numer   = k * acc - (a_norm * b_norm);
  auto Q_denom = k * a_sq_norm - (a_norm * a_norm);
  auto R_denom = k * b_sq_norm - (b_norm * b_norm);

  acc = 1 - (numer / raft::sqrt(Q_denom * R_denom));

  std::int64_t outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]        = acc;
}

template <typename DataType>
void naiveDistance(DataType* dist,
                   const DataType* x,
                   const DataType* y,
                   std::int64_t m,
                   std::int64_t n,
                   std::int64_t k,
                   cuvs::distance::DistanceType type,
                   bool isRowMajor,
                   DataType metric_arg = 2.0f,
                   cudaStream_t stream = 0)
{
  static const dim3 TPB(4, 256, 1);
  dim3 nblks(raft::ceildiv(m, (std::int64_t)TPB.x), raft::ceildiv(n, (std::int64_t)TPB.y), 1);

  switch (type) {
    case cuvs::distance::DistanceType::Canberra:
    case cuvs::distance::DistanceType::Linf:
    case cuvs::distance::DistanceType::L1:
      naiveL1_Linf_CanberraDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
    case cuvs::distance::DistanceType::L2Unexpanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2Expanded:
      naiveDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case cuvs::distance::DistanceType::CosineExpanded:
      naiveCosineDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case cuvs::distance::DistanceType::HellingerExpanded:
      naiveHellingerDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case cuvs::distance::DistanceType::LpUnexpanded:
      naiveLpUnexpDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor, metric_arg);
      break;
    case cuvs::distance::DistanceType::HammingUnexpanded:
      naiveHammingDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case cuvs::distance::DistanceType::InnerProduct:
      naiveInnerProductKernel<DataType><<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case cuvs::distance::DistanceType::JensenShannon:
      naiveJensenShannonDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case cuvs::distance::DistanceType::RusselRaoExpanded:
      naiveRussellRaoDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case cuvs::distance::DistanceType::KLDivergence:
      naiveKLDivergenceDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case cuvs::distance::DistanceType::CorrelationExpanded:
      naiveCorrelationDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    default: FAIL() << "should be here\n";
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataType>
struct DistanceInputs {
  DataType tolerance;
  std::int64_t m, n, k;
  bool isRowMajor;
  unsigned long long int seed;
  DataType metric_arg = 2.0f;
};

template <typename DataType>
::std::ostream& operator<<(::std::ostream& os, const DistanceInputs<DataType>& dims)
{
  return os;
}

// TODO: Remove when mdspan-based raft::runtime::distance::pairwise_distance is
// implemented.
//
// Context:
// https://github.com/rapidsai/raft/issues/1338
template <typename layout>
constexpr bool layout_to_row_major();

template <>
constexpr bool layout_to_row_major<raft::layout_c_contiguous>()
{
  return true;
}
template <>
constexpr bool layout_to_row_major<raft::layout_f_contiguous>()
{
  return false;
}

template <cuvs::distance::DistanceType distanceType, typename DataType, typename layout>
void distanceLauncher(raft::resources const& handle,
                      DataType* x,
                      DataType* y,
                      DataType* dist,
                      DataType* dist2,
                      std::int64_t m,
                      std::int64_t n,
                      std::int64_t k,
                      DistanceInputs<DataType>& params,
                      DataType threshold,
                      DataType metric_arg = 2.0f)
{
  auto x_v    = raft::make_device_matrix_view<DataType, std::int64_t, layout>(x, m, k);
  auto y_v    = raft::make_device_matrix_view<DataType, std::int64_t, layout>(y, n, k);
  auto dist_v = raft::make_device_matrix_view<DataType, std::int64_t, layout>(dist, m, n);

  cuvs::distance::pairwise_distance(handle, x_v, y_v, dist_v, distanceType, metric_arg);
}

template <cuvs::distance::DistanceType distanceType, typename DataType>
class DistanceTest : public ::testing::TestWithParam<DistanceInputs<DataType>> {
 public:
  DistanceTest()
    : params(::testing::TestWithParam<DistanceInputs<DataType>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      dist_ref(params.m * params.n, stream),
      dist(params.m * params.n, stream),
      dist2(params.m * params.n, stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::random::RngState r(params.seed);
    std::int64_t m      = params.m;
    std::int64_t n      = params.n;
    std::int64_t k      = params.k;
    DataType metric_arg = params.metric_arg;
    bool isRowMajor     = params.isRowMajor;
    if (distanceType == cuvs::distance::DistanceType::HellingerExpanded ||
        distanceType == cuvs::distance::DistanceType::JensenShannon ||
        distanceType == cuvs::distance::DistanceType::KLDivergence) {
      // Hellinger works only on positive numbers
      uniform(handle, r, x.data(), m * k, DataType(0.0), DataType(1.0));
      uniform(handle, r, y.data(), n * k, DataType(0.0), DataType(1.0));
    } else if (distanceType == cuvs::distance::DistanceType::RusselRaoExpanded) {
      uniform(handle, r, x.data(), m * k, DataType(0.0), DataType(1.0));
      uniform(handle, r, y.data(), n * k, DataType(0.0), DataType(1.0));
      // Russel rao works on boolean values.
      bernoulli(handle, r, x.data(), m * k, 0.5f);
      bernoulli(handle, r, y.data(), n * k, 0.5f);
    } else {
      uniform(handle, r, x.data(), m * k, DataType(-1.0), DataType(1.0));
      uniform(handle, r, y.data(), n * k, DataType(-1.0), DataType(1.0));
    }
    naiveDistance(
      dist_ref.data(), x.data(), y.data(), m, n, k, distanceType, isRowMajor, metric_arg, stream);

    DataType threshold = -10000.f;

    if (isRowMajor) {
      distanceLauncher<distanceType, DataType, raft::layout_c_contiguous>(handle,
                                                                          x.data(),
                                                                          y.data(),
                                                                          dist.data(),
                                                                          dist2.data(),
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          params,
                                                                          threshold,
                                                                          metric_arg);

    } else {
      distanceLauncher<distanceType, DataType, raft::layout_f_contiguous>(handle,
                                                                          x.data(),
                                                                          y.data(),
                                                                          dist.data(),
                                                                          dist2.data(),
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          params,
                                                                          threshold,
                                                                          metric_arg);
    }
    raft::resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  DistanceInputs<DataType> params;
  rmm::device_uvector<DataType> x, y, dist_ref, dist, dist2;
};

/*
 * This test suite verifies the path when X and Y are same buffer,
 * distance metrics which requires norms like L2 expanded/cosine/correlation
 * takes a more optimal path in such case to skip norm calculation for Y buffer.
 * It may happen that though both X and Y are same buffer but user passes
 * different dimensions for them like in case of tiled_brute_force_knn.
 */
template <cuvs::distance::DistanceType distanceType, typename DataType>
class DistanceTestSameBuffer : public ::testing::TestWithParam<DistanceInputs<DataType>> {
 public:
  using dev_vector = rmm::device_uvector<DataType>;
  DistanceTestSameBuffer()
    : params(::testing::TestWithParam<DistanceInputs<DataType>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      dist_ref({dev_vector(params.m * params.m, stream), dev_vector(params.m * params.m, stream)}),
      dist({dev_vector(params.m * params.m, stream), dev_vector(params.m * params.m, stream)}),
      dist2({dev_vector(params.m * params.m, stream), dev_vector(params.m * params.m, stream)})
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::random::RngState r(params.seed);
    std::int64_t m      = params.m;
    std::int64_t n      = params.m;
    std::int64_t k      = params.k;
    DataType metric_arg = params.metric_arg;
    bool isRowMajor     = params.isRowMajor;
    if (distanceType == cuvs::distance::DistanceType::HellingerExpanded ||
        distanceType == cuvs::distance::DistanceType::JensenShannon ||
        distanceType == cuvs::distance::DistanceType::KLDivergence) {
      // Hellinger works only on positive numbers
      uniform(handle, r, x.data(), m * k, DataType(0.0), DataType(1.0));
    } else if (distanceType == cuvs::distance::DistanceType::RusselRaoExpanded) {
      uniform(handle, r, x.data(), m * k, DataType(0.0), DataType(1.0));
      // Russel rao works on boolean values.
      bernoulli(handle, r, x.data(), m * k, 0.5f);
    } else {
      uniform(handle, r, x.data(), m * k, DataType(-1.0), DataType(1.0));
    }

    for (std::int64_t i = 0; i < 2; i++) {
      // both X and Y are same buffer but when i = 1
      // different dimensions for x & y is passed.
      m = m / (i + 1);
      naiveDistance(dist_ref[i].data(),
                    x.data(),
                    x.data(),
                    m,
                    n,
                    k,
                    distanceType,
                    isRowMajor,
                    metric_arg,
                    stream);

      DataType threshold = -10000.f;

      if (isRowMajor) {
        distanceLauncher<distanceType, DataType, raft::layout_c_contiguous>(handle,
                                                                            x.data(),
                                                                            x.data(),
                                                                            dist[i].data(),
                                                                            dist2[i].data(),
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            params,
                                                                            threshold,
                                                                            metric_arg);

      } else {
        distanceLauncher<distanceType, DataType, raft::layout_f_contiguous>(handle,
                                                                            x.data(),
                                                                            x.data(),
                                                                            dist[i].data(),
                                                                            dist2[i].data(),
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            params,
                                                                            threshold,
                                                                            metric_arg);
      }
    }
    raft::resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  DistanceInputs<DataType> params;
  dev_vector x;
  static const std::int64_t N = 2;
  std::array<dev_vector, N> dist_ref, dist, dist2;
};

template <cuvs::distance::DistanceType distanceType>
class BigMatrixDistanceTest : public ::testing::Test {
 public:
  BigMatrixDistanceTest()
    : x(m * k, raft::resource::get_cuda_stream(handle)),
      dist(std::size_t(m) * m, raft::resource::get_cuda_stream(handle)){};
  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    constexpr float metric_arg = 0.0f;
    auto x_v =
      raft::make_device_matrix_view<float, std::int64_t, raft::layout_c_contiguous>(x.data(), m, k);
    auto dist_v = raft::make_device_matrix_view<float, std::int64_t, raft::layout_c_contiguous>(
      dist.data(), m, n);

    cuvs::distance::pairwise_distance(handle, x_v, x_v, dist_v, distanceType, metric_arg);
    raft::resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  std::int64_t m = 48000;
  std::int64_t n = 48000;
  std::int64_t k = 1;
  rmm::device_uvector<float> x, dist;
};
}  // end namespace distance
}  // namespace cuvs
