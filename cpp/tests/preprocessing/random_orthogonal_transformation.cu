/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include <cuvs/preprocessing/linear_transform/random_orthogonal.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/stats/stddev.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cuvs::preprocessing::linear_transform::random_orthogonal {

template <typename T>
struct TransformationInputs {
  int rows;
  int cols;
};

template <class T>
constexpr double error_threshold_const = 0;
template <>
constexpr double error_threshold_const<double> = 1e-14;
template <>
constexpr double error_threshold_const<float> = 5e-6;
template <>
constexpr double error_threshold_const<half> = 5e-3;
template <class T>
double error_threshold(const T dim)
{
  return error_threshold_const<T> * sqrt(static_cast<double>(dim));
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const TransformationInputs<T>& inputs)
{
  return os << " rows:" << inputs.rows << " cols:" << inputs.cols;
}

template <typename T>
class RandomOrthogonalTransformation : public ::testing::TestWithParam<TransformationInputs<T>> {
 public:
  RandomOrthogonalTransformation()
    : params_(::testing::TestWithParam<TransformationInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      input_(0, stream)
  {
  }

  double getRelativeErrorStddev(const T* array_a, const T* array_b, size_t size, float quantile)
  {
    // relative error elementwise
    rmm::device_uvector<double> relative_error(size, stream);
    raft::linalg::binaryOp(
      relative_error.data(),
      array_a,
      array_b,
      size,
      [] __device__(double a, double b) {
        return a != b ? (raft::abs(a - b) / raft::max(raft::abs(a), raft::abs(b))) : 0;
      },
      stream);

    // sort by size --> remove largest errors to account for quantile chosen
    thrust::sort(raft::resource::get_thrust_policy(handle),
                 relative_error.data(),
                 relative_error.data() + size);
    int elements_to_consider =
      std::ceil(double(params_.quantization_params.quantile) * double(size));

    rmm::device_uvector<double> mu(1, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(mu.data(), 0, sizeof(double), stream));

    rmm::device_uvector<double> error_stddev(1, stream);
    raft::stats::stddev<true>(error_stddev.data(),
                              relative_error.data(),
                              mu.data(),
                              1,
                              elements_to_consider,
                              false,
                              stream);

    double error_stddev_h;
    raft::update_host(&error_stddev_h, error_stddev.data(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    return error_stddev_h;
  }

 protected:
  void testRandomOrthogonalTransformation()
  {
    // dataset identical on host / device
    auto dataset = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto dataset_h = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(host_input_.data()), rows_, cols_);

    size_t print_size = std::min(input_.size(), 20ul);

    cuvs::preprocessing::linear_transform::random_orthogonal::params params;
    auto transformer =
      cuvs::preprocessing::linear_transform::random_orthogonal::train(handle, params, dataset);

    {
      auto transformed_input_h = raft::make_host_matrix<T, int64_t>(rows_, cols_);
      auto transformed_input_d = raft::make_device_matrix<T, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::linear_transform::random_orthogonal::transform(
        handle, transformer, dataset, transformed_input_d.view());
      cuvs::preprocessing::linear_transform::random_orthogonal::transform(
        handle, transformer, dataset_h, transformed_input_h.view());

      // test transform host/device equal
      ASSERT_TRUE(devArrMatchHost(transformed_input_h.data_handle(),
                                  transformed_input_d.data_handle(),
                                  input_.size(),
                                  cuvs::CompareApprox<T>(error_threshold<T>(cols_)),
                                  stream));

      auto transformed_input_h_const_view = raft::make_host_matrix_view<const T, int64_t>(
        transformed_input_h.data_handle(), rows_, cols_);
      auto re_transformed_input_h = raft::make_host_matrix<T, int64_t>(rows_, cols_);
      cuvs::preprocessing::linear_transform::random_orthogonal::inverse_transform(
        handle, transformer, transformed_input_h_const_view, re_transformed_input_h.view());

      auto transformed_input_d_const_view = raft::make_device_matrix_view<const T, int64_t>(
        transformed_input_d.data_handle(), rows_, cols_);
      auto re_transformed_input_d = raft::make_device_matrix<T, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::linear_transform::random_orthogonal::inverse_transform(
        handle, transformer, transformed_input_d_const_view, re_transformed_input_d.view());

      // test transform host/device equal
      ASSERT_TRUE(
        devArrMatchHost(re_transformed_input_h.data_handle(),
                        re_transformed_input_d.data_handle(),
                        input_.size(),
                        cuvs::CompareApprox<T>(error_threshold<T>(cols_) * 2 /*=transform+inv*/),
                        stream));
      ASSERT_TRUE(
        devArrMatchHost(re_transformed_input_h.data_handle(),
                        dataset.data_handle(),
                        input_.size(),
                        cuvs::CompareApprox<T>(error_threshold<T>(cols_) * 2 /*=transform+inv*/),
                        stream));
    }
  }

  void SetUp() override
  {
    rows_ = params_.rows;
    cols_ = params_.cols;

    int n_elements = rows_ * cols_;
    input_.resize(n_elements, stream);
    host_input_.resize(n_elements);

    // random input
    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);
    uniform(handle, r, input_.data(), input_.size(), static_cast<T>(-8), static_cast<T>(8));

    raft::update_host(host_input_.data(), input_.data(), input_.size(), stream);

    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  TransformationInputs<T> params_;
  int rows_;
  int cols_;
  rmm::device_uvector<T> input_;
  std::vector<T> host_input_;
};

template <typename T>
const std::vector<TransformationInputs<T>> inputs = {{1000, 1},
                                                     {1000, 3},
                                                     {1000, 13},
                                                     {1000, 128},
                                                     {1000, 199},
                                                     {1000, 876},
                                                     {1000, 1289},
                                                     {10000, 67},
                                                     {10000, 128}};

typedef RandomOrthogonalTransformation<float> RandomOrthogonalTransformation_float_int8t;
TEST_P(RandomOrthogonalTransformation_float_int8t, RandomOrthogonalTransformationTest)
{
  this->testRandomOrthogonalTransformation();
}

typedef RandomOrthogonalTransformation<double> RandomOrthogonalTransformation_double_int8t;
TEST_P(RandomOrthogonalTransformation_double_int8t, RandomOrthogonalTransformationTest)
{
  this->testRandomOrthogonalTransformation();
}

typedef RandomOrthogonalTransformation<half> RandomOrthogonalTransformation_half_int8t;
TEST_P(RandomOrthogonalTransformation_half_int8t, RandomOrthogonalTransformationTest)
{
  this->testRandomOrthogonalTransformation();
}

INSTANTIATE_TEST_CASE_P(RandomOrthogonalTransformation,
                        RandomOrthogonalTransformation_float_int8t,
                        ::testing::ValuesIn(inputs<float>));
INSTANTIATE_TEST_CASE_P(RandomOrthogonalTransformation,
                        RandomOrthogonalTransformation_double_int8t,
                        ::testing::ValuesIn(inputs<double>));
INSTANTIATE_TEST_CASE_P(RandomOrthogonalTransformation,
                        RandomOrthogonalTransformation_half_int8t,
                        ::testing::ValuesIn(inputs<half>));

}  // namespace cuvs::preprocessing::linear_transform::random_orthogonal
