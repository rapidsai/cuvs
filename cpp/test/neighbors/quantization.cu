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
#include <cuvs/neighbors/quantization.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cuvs::neighbors::quantization {

template <typename T>
struct QuantizationInputs {
  cuvs::neighbors::quantization::params quantization_params;
  int rows;
  int cols;
  T min = T(-1.0);
  T max = T(1.0);
};

std::ostream& operator<<(std::ostream& os, const params& params)
{
  return os << "quantile:" << params.quantile << " is_computed:" << params.is_computed
            << " min:" << params.min << " max:" << params.max << " scalar:" << params.scalar;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const QuantizationInputs<T>& inputs)
{
  return os << "quantization_params:<" << inputs.quantization_params << "> rows:" << inputs.rows
            << " cols:" << inputs.cols << " min:" << (double)inputs.min
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

 protected:
  void testScalarQuantization()
  {
    auto quantization_params_copy = params_.quantization_params;

    auto dataset = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto quantized_input =
      cuvs::neighbors::quantization::scalar_quantize(handle, params_.quantization_params, dataset);

    // check quantization params
    ASSERT_TRUE(params_.quantization_params.is_computed);
    std::cerr << "Params " << params_ << std::endl;
    if (input_.size() < 100) {
      raft::print_device_vector("Input array: ", input_.data(), input_.size(), std::cerr);

      rmm::device_uvector<int> quantization_for_print(input_.size(), stream);
      raft::linalg::unaryOp(quantization_for_print.data(),
                            quantized_input.data_handle(),
                            input_.size(),
                            raft::cast_op<int>{},
                            stream);
      raft::resource::sync_stream(handle, stream);
      raft::print_device_vector(
        "Quantized array: ", quantization_for_print.data(), input_.size(), std::cerr);
    }

    // check second run same result (no quantile computation)
    auto quantized_input2 =
      cuvs::neighbors::quantization::scalar_quantize(handle, params_.quantization_params, dataset);
    ASSERT_TRUE(devArrMatch(quantized_input.data_handle(),
                            quantized_input2.data_handle(),
                            input_.size(),
                            cuvs::Compare<QuantI>(),
                            stream));

    // sort_by_key (input, quantization) -- check <= on result
    thrust::sort_by_key(raft::resource::get_thrust_policy(handle),
                        input_.data(),
                        input_.data() + input_.size(),
                        quantized_input2.data_handle());
    std::vector<QuantI> quantized_input_sorted_host(input_.size());
    raft::update_host(
      quantized_input_sorted_host.data(), quantized_input2.data_handle(), input_.size(), stream);
    raft::resource::sync_stream(handle, stream);

    for (size_t i = 0; i < input_.size() - 1; ++i) {
      ASSERT_TRUE(quantized_input_sorted_host[i] <= quantized_input_sorted_host[i + 1]);
    }

    // compare with host
    auto dataset_h = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(host_input_.data()), rows_, cols_);
    {
      // note that this will utilize the scalar computed via the previous (device) call
      auto quantized_input_host_1 = cuvs::neighbors::quantization::scalar_quantize(
        handle, params_.quantization_params, dataset_h);
      ASSERT_TRUE(devArrMatchHost(quantized_input_host_1.data_handle(),
                                  quantized_input.data_handle(),
                                  input_.size(),
                                  cuvs::Compare<QuantI>(),
                                  stream));

      // re-evaluate scalar
      auto quantized_input_host_2 =
        cuvs::neighbors::quantization::scalar_quantize(handle, quantization_params_copy, dataset_h);
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

// TODO -- host first or last?
// can always test last but also need to run sampling ...

template <typename T>
const std::vector<QuantizationInputs<T>> inputs = {
  {{0.99}, 10, 20, T(0.0), T(1.0)},
  {{0.95}, 100, 2000, T(-500.0), T(100.0)},
  {{0.59}, 100, 200},
  {{0.94}, 10, 20, T(-1.0), T(0.0)},
  {{0.95}, 10, 2, T(50.0), T(100.0)},
  {{0.95}, 10, 20, T(-500.0), T(-100.0)},
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

}  // namespace cuvs::neighbors::quantization
