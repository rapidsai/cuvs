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
#include <cuvs/preprocessing/quantize/binary.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/stats/stddev.cuh>
#include <thrust/reduce.h>

namespace cuvs::preprocessing::quantize::binary {

template <typename T>
struct BinaryQuantizationInputs {
  int rows;
  int cols;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const BinaryQuantizationInputs<T>& inputs)
{
  return os << "> rows:" << inputs.rows << " cols:" << inputs.cols;
}

template <typename T, typename QuantI>
class BinaryQuantizationTest : public ::testing::TestWithParam<BinaryQuantizationInputs<T>> {
 public:
  BinaryQuantizationTest()
    : params_(::testing::TestWithParam<BinaryQuantizationInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      input_(0, stream)
  {
  }

 protected:
  void testBinaryQuantization()
  {
    // dataset identical on host / device
    auto dataset = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto dataset_h = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(host_input_.data()), rows_, cols_);

    {
      static_assert(std::is_same_v<QuantI, uint8_t>);

      cuvs::preprocessing::quantize::binary::params params;

      const auto col_quantized = raft::div_rounding_up_safe(cols_, 8);
      auto quantized_input_h   = raft::make_host_matrix<QuantI, int64_t>(rows_, cols_);
      auto quantized_input_d   = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::quantize::binary::transform(
        handle, params, dataset, quantized_input_d.view());
      cuvs::preprocessing::quantize::binary::transform(
        handle, params, dataset_h, quantized_input_h.view());

      ASSERT_TRUE(devArrMatchHost(quantized_input_h.data_handle(),
                                  quantized_input_d.data_handle(),
                                  input_.size(),
                                  cuvs::Compare<QuantI>(),
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
    uniform(handle, r, input_.data(), input_.size(), static_cast<T>(-1), static_cast<T>(1));

    raft::update_host(host_input_.data(), input_.data(), input_.size(), stream);

    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  BinaryQuantizationInputs<T> params_;
  int rows_;
  int cols_;
  rmm::device_uvector<T> input_;
  std::vector<T> host_input_;
};

template <typename T>
const std::vector<BinaryQuantizationInputs<T>> inputs = {
  {5, 5},
  {100, 7},
  {100, 128},
  {100, 1999},
  {1000, 1999},
};

typedef BinaryQuantizationTest<float, uint8_t> QuantizationTest_float_uint8t;
TEST_P(QuantizationTest_float_uint8t, BinaryQuantizationTest) { this->testBinaryQuantization(); }

typedef BinaryQuantizationTest<double, uint8_t> QuantizationTest_double_uint8t;
TEST_P(QuantizationTest_double_uint8t, BinaryQuantizationTest) { this->testBinaryQuantization(); }

typedef BinaryQuantizationTest<half, uint8_t> QuantizationTest_half_uint8t;
TEST_P(QuantizationTest_half_uint8t, BinaryQuantizationTest) { this->testBinaryQuantization(); }

INSTANTIATE_TEST_CASE_P(BinaryQuantizationTest,
                        QuantizationTest_float_uint8t,
                        ::testing::ValuesIn(inputs<float>));
INSTANTIATE_TEST_CASE_P(BinaryQuantizationTest,
                        QuantizationTest_double_uint8t,
                        ::testing::ValuesIn(inputs<double>));
INSTANTIATE_TEST_CASE_P(BinaryQuantizationTest,
                        QuantizationTest_half_uint8t,
                        ::testing::ValuesIn(inputs<half>));

}  // namespace cuvs::preprocessing::quantize::binary
