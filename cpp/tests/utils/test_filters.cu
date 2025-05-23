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

#include <cuvs/core/bitset.hpp>
#include <cuvs/utils/filters.hpp>
#include <faiss/impl/IDSelector.h>
#include <raft/core/bitset.cuh>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/linalg/init.cuh>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>

namespace faiss {
IDSelectorRange::IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted)
  : imin(imin), imax(imax), assume_sorted(assume_sorted)
{
}

bool IDSelectorRange::is_member(idx_t id) const { return id >= imin && id < imax; }
}  // namespace faiss

namespace cuvs::utils {

struct test_spec_convert {
  uint64_t bitset_len;
};
auto operator<<(std::ostream& os, const test_spec_convert& ss) -> std::ostream&
{
  os << "convert{bitset_len: " << ss.bitset_len << "}";
  return os;
}

template <typename bitset_t, typename index_t>
struct IDSelectorTest : public testing::TestWithParam<test_spec_convert> {
  explicit IDSelectorTest() : spec(testing::TestWithParam<test_spec_convert>::GetParam()) {}

  void run()
  {
    // IDSelectorRange
    {
      // take random imin and imax, check all ids
      std::srand(std::time(0));
      auto imin = std::rand() % spec.bitset_len;
      auto imax = std::rand() % spec.bitset_len;
      if (imin > imax) std::swap(imin, imax);
      auto selector = faiss::IDSelectorRange(imin, imax);
      auto bitset   = cuvs::core::bitset<bitset_t, index_t>(res, spec.bitset_len, false);
      auto nbits    = sizeof(bitset_t) * 8;

      cuvs::utils::convert_to_bitset(res, selector, bitset.view());
      auto bitset_converted_cpu = raft::make_host_vector<bitset_t, index_t>(bitset.n_elements());
      raft::copy(res, bitset_converted_cpu.view(), bitset.to_mdspan());
      raft::resource::sync_stream(res);
      auto bitset_view_cpu = cuvs::core::bitset_view<bitset_t, index_t>(
        bitset_converted_cpu.data_handle(), spec.bitset_len);
      for (uint64_t i = 0; i < spec.bitset_len; i++) {
        if (bitset_view_cpu.test(i) != selector.is_member(i)) {
          ASSERT_TRUE(testing::AssertionFailure()
                      << "actual=" << bitset_view_cpu.test(i)
                      << " != expected=" << selector.is_member(i) << " @" << i
                      << " bitset_len: " << spec.bitset_len << " imin: " << imin
                      << " imax: " << imax << " bit_element: " << bitset_converted_cpu(i / nbits));
        }
      }
    }
  }
  raft::resources res;
  const test_spec_convert spec;
};

auto inputs_convert =
  ::testing::Values(test_spec_convert{100}, test_spec_convert{1000}, test_spec_convert{10000});
/*
using Uint8_32 = IDSelectorTest<uint8_t, uint32_t>;
TEST_P(Uint8_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(IDSelectorTest, Uint8_32, inputs_convert);
*/
using Uint32_32 = IDSelectorTest<uint32_t, uint32_t>;
TEST_P(Uint32_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(IDSelectorTest, Uint32_32, inputs_convert);

/*
using Uint64_32 = IDSelectorTest<uint64_t, uint32_t>;
TEST_P(Uint64_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(IDSelectorTest, Uint64_32, inputs_convert);
*/

}  // namespace cuvs::utils
