/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "test_utils.cuh"
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <raft/core/host_mdarray.hpp>

#include "../../src/core/interop.hpp"

#include <gtest/gtest.h>
#include <sys/types.h>

namespace cuvs::core {

TEST(Interop, FromDLPack)
{
  raft::resources res;
  auto data = raft::make_host_vector<float>(res, 2);
  data(0)   = 5;
  data(1)   = 10;

  auto device    = DLDevice{kDLCPU};
  auto data_type = DLDataType{kDLFloat, 4 * 8, 1};
  auto shape     = std::vector<int64_t>{2};

  auto tensor         = DLTensor{data.data_handle(), device, 1, data_type, shape.data()};
  auto managed_tensor = DLManagedTensor{tensor};

  using mdspan_type = raft::host_mdspan<float const, raft::vector_extent<int64_t>>;
  auto out          = from_dlpack<mdspan_type>(&managed_tensor);

  ASSERT_EQ(out.rank(), data.rank());
  ASSERT_EQ(out.extent(0), data.extent(0));
  ASSERT_EQ(out(0), data(0));
  ASSERT_EQ(out(1), data(1));
}

TEST(Interop, FromDLPackStrides)
{
  raft::resources res;
  auto data = raft::make_host_matrix<float>(res, 3, 2);
  data(0, 0)   = 1;
  data(1, 0)   = 2;
  data(2, 0)   = 3;
  data(0, 1)   = 4;
  data(1, 1)   = 5;
  data(2, 1)   = 6;

  auto device    = DLDevice{kDLCPU};
  auto data_type = DLDataType{kDLFloat, 4 * 8, 1};
  auto shape     = std::vector<int64_t>{3 ,2};

  auto tensor         = DLTensor{data.data_handle(), device, 2, data_type, shape.data()};
  auto managed_tensor = DLManagedTensor{tensor};

  // converting a 2D dltensor to a 1D mspan should fail
  using vector_mdspan_type = raft::host_mdspan<float const, raft::vector_extent<int64_t>>;
  ASSERT_THROW(from_dlpack<vector_mdspan_type>(&managed_tensor), raft::logic_error);

  // No stride information in the dltensor indicates row major
  using mdspan_type = raft::host_matrix_view<float const, int64_t, raft::row_major>;
  auto out = from_dlpack<mdspan_type>(&managed_tensor);
  ASSERT_EQ(out.rank(), data.rank());
  ASSERT_EQ(out.extent(0), data.extent(0));
  ASSERT_EQ(out.extent(1), data.extent(1));
  for (int64_t row = 0; row < data.extent(0); row++) {
    for (int64_t col = 0; col < data.extent(1); col++) {
      ASSERT_EQ(out(row, col), data(row, col));
    }
  }

  // asking for a col-major mdspan should fail if no strides are present
  using colmajor_mdspan_type = raft::host_matrix_view<float const, int64_t, raft::col_major>;
  ASSERT_THROW(from_dlpack<colmajor_mdspan_type>(&managed_tensor), raft::logic_error);

  // Setting strides equal to row major should also work
  auto strides = std::vector<int64_t>{2, 1};
  managed_tensor.dl_tensor.strides = strides.data();
  auto out_strided = from_dlpack<mdspan_type>(&managed_tensor);
  ASSERT_EQ(out_strided.rank(), data.rank());

  // Setting strides indicating col-major should be able to convert to a col-major
  // mdspan
  auto strides_colmajor = std::vector<int64_t>{1, 3};
  managed_tensor.dl_tensor.strides = strides_colmajor.data();
  auto out_colmajor = from_dlpack<colmajor_mdspan_type>(&managed_tensor);
  ASSERT_EQ(out_colmajor.rank(), data.rank());

  // But shouldn't be able to convert to a row-major
  ASSERT_THROW(from_dlpack<mdspan_type>(&managed_tensor), raft::logic_error);
}

}  // namespace cuvs::core
