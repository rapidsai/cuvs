/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/stats/stddev.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cuvs::preprocessing::quantize::scalar {

template <typename T>
struct QuantizationInputs {
  cuvs::preprocessing::quantize::scalar::params quantization_params;
  int rows;
  int cols;
  T min            = T(-1.0);
  T max            = T(1.0);
  double threshold = 2e-2;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const QuantizationInputs<T>& inputs)
{
  return os << "quantization_quantile:<" << inputs.quantization_params.quantile
            << "> rows:" << inputs.rows << " cols:" << inputs.cols << " min:" << (double)inputs.min
            << " max:" << (double)inputs.max;
}

template <typename T, typename QuantI>
class QuantizationTest : public ::testing::TestWithParam<QuantizationInputs<T>> {
 public:
  QuantizationTest()
    : params_(::testing::TestWithParam<QuantizationInputs<T>>::GetParam()),
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
    raft::stats::stddev(error_stddev.data(),
                        relative_error.data(),
                        mu.data(),
                        1,
                        elements_to_consider,
                        false,
                        true,
                        stream);

    double error_stddev_h;
    raft::update_host(&error_stddev_h, error_stddev.data(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    return error_stddev_h;
  }

 protected:
  void testScalarQuantization()
  {
    // dataset identical on host / device
    auto dataset = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto dataset_h = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(host_input_.data()), rows_, cols_);

    size_t print_size = std::min(input_.size(), 20ul);

    // train quantizer_1 on device
    auto quantizer_1 =
      cuvs::preprocessing::quantize::scalar::train(handle, params_.quantization_params, dataset);
    std::cerr << "Q1: min = " << (double)quantizer_1.min_ << ", max = " << (double)quantizer_1.max_
              << std::endl;

    {
      auto quantized_input_h = raft::make_host_matrix<QuantI, int64_t>(rows_, cols_);
      auto quantized_input_d = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_1, dataset, quantized_input_d.view());
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_1, dataset_h, quantized_input_h.view());

      {
        raft::print_device_vector("Input array: ", input_.data(), print_size, std::cerr);

        rmm::device_uvector<int> quantization_for_print(print_size, stream);
        raft::linalg::unaryOp(quantization_for_print.data(),
                              quantized_input_d.data_handle(),
                              print_size,
                              raft::cast_op<int>{},
                              stream);
        raft::resource::sync_stream(handle, stream);
        raft::print_device_vector(
          "Quantized array 1: ", quantization_for_print.data(), print_size, std::cerr);
      }

      // test (inverse) transform host/device equal
      ASSERT_TRUE(devArrMatchHost(quantized_input_h.data_handle(),
                                  quantized_input_d.data_handle(),
                                  input_.size(),
                                  cuvs::Compare<QuantI>(),
                                  stream));

      auto quantized_input_h_const_view = raft::make_host_matrix_view<const QuantI, int64_t>(
        quantized_input_h.data_handle(), rows_, cols_);
      auto re_transformed_input_h = raft::make_host_matrix<T, int64_t>(rows_, cols_);
      cuvs::preprocessing::quantize::scalar::inverse_transform(
        handle, quantizer_1, quantized_input_h_const_view, re_transformed_input_h.view());

      auto quantized_input_d_const_view = raft::make_device_matrix_view<const QuantI, int64_t>(
        quantized_input_d.data_handle(), rows_, cols_);
      auto re_transformed_input_d = raft::make_device_matrix<T, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::quantize::scalar::inverse_transform(
        handle, quantizer_1, quantized_input_d_const_view, re_transformed_input_d.view());
      raft::print_device_vector(
        "re-transformed array: ", re_transformed_input_d.data_handle(), print_size, std::cerr);

      {
        double l2_error = getRelativeErrorStddev(dataset.data_handle(),
                                                 re_transformed_input_d.data_handle(),
                                                 input_.size(),
                                                 params_.quantization_params.quantile);
        std::cerr << "error stddev = " << l2_error << ", threshold = " << params_.threshold
                  << std::endl;
        // test (inverse) transform close to original dataset
        ASSERT_TRUE(l2_error < params_.threshold);
      }
    }

    // train quantizer_2 on host
    auto quantizer_2 =
      cuvs::preprocessing::quantize::scalar::train(handle, params_.quantization_params, dataset_h);
    std::cerr << "Q2: min = " << (double)quantizer_2.min_ << ", max = " << (double)quantizer_2.max_
              << std::endl;

    // check both quantizers are the same (valid if sampling is identical)
    if (input_.size() <= 1000000) {
      ASSERT_TRUE((double)quantizer_1.min_ == (double)quantizer_2.min_);
      ASSERT_TRUE((double)quantizer_1.max_ == (double)quantizer_2.max_);
    }

    {
      // test transform host/device equal
      auto quantized_input_h = raft::make_host_matrix<QuantI, int64_t>(rows_, cols_);
      auto quantized_input_d = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_2, dataset, quantized_input_d.view());
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_2, dataset_h, quantized_input_h.view());

      {
        rmm::device_uvector<int> quantization_for_print(print_size, stream);
        raft::linalg::unaryOp(quantization_for_print.data(),
                              quantized_input_d.data_handle(),
                              print_size,
                              raft::cast_op<int>{},
                              stream);
        raft::resource::sync_stream(handle, stream);
        raft::print_device_vector(
          "Quantized array 2: ", quantization_for_print.data(), print_size, std::cerr);
      }

      ASSERT_TRUE(devArrMatchHost(quantized_input_h.data_handle(),
                                  quantized_input_d.data_handle(),
                                  input_.size(),
                                  cuvs::Compare<QuantI>(),
                                  stream));
    }

    // sort_by_key (input, quantization) -- check <= on result
    {
      auto quantized_input = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_1, dataset, quantized_input.view());
      thrust::sort_by_key(raft::resource::get_thrust_policy(handle),
                          input_.data(),
                          input_.data() + input_.size(),
                          quantized_input.data_handle());
      std::vector<QuantI> quantized_input_sorted_host(input_.size());
      raft::update_host(
        quantized_input_sorted_host.data(), quantized_input.data_handle(), input_.size(), stream);
      raft::resource::sync_stream(handle, stream);

      for (size_t i = 0; i < input_.size() - 1; ++i) {
        ASSERT_TRUE(quantized_input_sorted_host[i] <= quantized_input_sorted_host[i + 1]);
      }
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
    uniform(handle, r, input_.data(), input_.size(), params_.min, params_.max);

    raft::update_host(host_input_.data(), input_.data(), input_.size(), stream);

    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  QuantizationInputs<T> params_;
  int rows_;
  int cols_;
  rmm::device_uvector<T> input_;
  std::vector<T> host_input_;
};

template <typename T>
const std::vector<QuantizationInputs<T>> inputs = {
  {{1.0}, 5, 5, T(0.0), T(1.0)},
  {{0.98}, 10, 20, T(0.0), T(1.0)},
  {{0.90}, 1000, 1500, T(-500.0), T(100.0)},
  {{0.59}, 100, 200},
  {{0.1}, 1, 1, T(0.0), T(1.0)},
  {{0.01}, 50, 50, T(0.0), T(1.0)},
  {{0.94}, 10, 20, T(-1.0), T(0.0)},
  {{0.95}, 10, 2, T(50.0), T(100.0)},
  {{0.95}, 10, 20, T(-500.0), T(-100.0)},
  {{0.95}, 10, 20, T(5.0), T(5.0)},
};

typedef QuantizationTest<float, int8_t> QuantizationTest_float_int8t;
TEST_P(QuantizationTest_float_int8t, ScalarQuantizationTest) { this->testScalarQuantization(); }

typedef QuantizationTest<double, int8_t> QuantizationTest_double_int8t;
TEST_P(QuantizationTest_double_int8t, ScalarQuantizationTest) { this->testScalarQuantization(); }

typedef QuantizationTest<half, int8_t> QuantizationTest_half_int8t;
TEST_P(QuantizationTest_half_int8t, ScalarQuantizationTest) { this->testScalarQuantization(); }

INSTANTIATE_TEST_CASE_P(QuantizationTest,
                        QuantizationTest_float_int8t,
                        ::testing::ValuesIn(inputs<float>));
INSTANTIATE_TEST_CASE_P(QuantizationTest,
                        QuantizationTest_double_int8t,
                        ::testing::ValuesIn(inputs<double>));
INSTANTIATE_TEST_CASE_P(QuantizationTest,
                        QuantizationTest_half_int8t,
                        ::testing::ValuesIn(inputs<half>));

}  // namespace cuvs::preprocessing::quantize::scalar
