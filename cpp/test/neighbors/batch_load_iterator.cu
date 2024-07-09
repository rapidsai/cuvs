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
#include "ann_utils.cuh"

#include <raft/core/copy.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace cuvs::spatial::knn::detail::utils {
template <typename DataT, typename IdxT>
struct AnnBatchLoadIteratorInputs {
  IdxT n_rows;
  IdxT row_width;
  IdxT batch_size;
  bool host_dataset;
  typename batch_load_iterator<DataT>::PrefetchOption prefetch;
};

template <typename DataT, typename IdxT>
class AnnBatchLoadIteratorTest : public ::testing::TestWithParam<AnnBatchLoadIteratorInputs<DataT, IdxT>> {
 public:
  AnnBatchLoadIteratorTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnBatchLoadIteratorInputs<DataT, IdxT>>::GetParam()),
      dataset(0, stream_)
  {
  }

  void testBatchedCopy()
  {
    batch_load_iterator<DataT> batches(ps.host_dataset ? host_dataset.data() : dataset.data(),
                                       ps.n_rows,
                                       ps.row_width,
                                       ps.batch_size,
                                       stream_,
                                       raft::resource::get_workspace_resource(handle_),
                                       ps.prefetch);
    IdxT batch_id = 0;
    if (ps.prefetch) { batches.prefetch_next_batch(); }
    for (auto& batch : batches) {
      raft::resource::sync_stream(handle_);
      IdxT cur_batch_size = std::min(ps.batch_size, ps.n_rows - batch_id * ps.batch_size);
      if (ps.host_dataset) {
        ASSERT_TRUE(devArrMatchHost(host_dataset.data() + batch_id * ps.row_width,
                                    batch.data(),
                                    cur_batch_size * ps.row_width,
                                    Compare<bool>(),
                                    stream_));
      } else {
        ASSERT_TRUE(devArrMatch(dataset.data() + batch_id * ps.row_width,
                                batch.data(),
                                cur_batch_size * ps.row_width,
                                Compare<bool>(),
                                stream_));
      }
      if (ps.prefetch) { batches.prefetch_next_batch(); }
      batch_id++;
    }
  }

  void SetUp() override
  {
    dataset.resize(ps.n_rows * ps.row_width, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, dataset.data(), ps.n_rows * ps.row_width, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, dataset.data(), ps.n_rows * ps.row_width, DataT(1), DataT(20));
    }
    host_dataset.resize(ps.n_rows * ps.row_width);
    raft::copy(host_dataset.data(), dataset.data(), ps.n_rows * ps.row_width, stream_);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    dataset.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnBatchLoadIteratorInputs<DataT, IdxT> ps;
  rmm::device_uvector<DataT> dataset;
  std::vector<DataT> host_dataset;
};

const std::vector<AnnBatchLoadIteratorInputs<float, int64_t>> inputs = {
  // test device input, batch iterator should directly
  {10000, 8, 65536, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {100000, 8, 65536, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {100000, 16, 65536, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {1000000, 8, 65536, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 8, 65536, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 8, 524288, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 16, 524288, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  // test host input without prefetching, batch iterator requires pageable copy
  {10000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {100000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {100000, 16, 65536, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {1000000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 8, 524288, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 16, 524288, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  // test host input with prefetching, batch iterator requires pageable copy
  {10000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {100000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {100000, 16, 65536, false, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {1000000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 8, 65536, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 8, 524288, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE},
  {10000000, 16, 524288, true, batch_load_iterator<float>::PrefetchOption::PREFETCH_NONE}};

typedef AnnBatchLoadIteratorTest<float, std::int64_t> AnnBatchLoadIteratorTest_float;
TEST_P(AnnBatchLoadIteratorTest_float, AnnBatchLoadIterator) { this->testBatchedCopy(); }
INSTANTIATE_TEST_CASE_P(AnnBatchLoadIteratorTest,
                        AnnBatchLoadIteratorTest_float,
                        ::testing::ValuesIn(inputs));
}  // namespace cuvs::spatial::knn::detail::utils