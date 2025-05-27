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
#include <cuvs/util/filters.hpp>
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
#include <random>

namespace faiss {
// Implementation copied from faiss/impl/IDSelector.cpp instead of linking to the library

IDSelectorRange::IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted)
  : imin(imin), imax(imax), assume_sorted(assume_sorted)
{
}

bool IDSelectorRange::is_member(idx_t id) const { return id >= imin && id < imax; }

IDSelectorArray::IDSelectorArray(size_t n, const idx_t* ids) : n(n), ids(ids) {}

bool IDSelectorArray::is_member(idx_t id) const
{
  for (size_t i = 0; i < n; i++) {
    if (ids[i] == id) return true;
  }
  return false;
}

IDSelectorBitmap::IDSelectorBitmap(size_t n, const uint8_t* bitmap) : n(n), bitmap(bitmap) {}

bool IDSelectorBitmap::is_member(idx_t ii) const
{
  uint64_t i = ii;
  if ((i >> 3) >= n) { return false; }
  return (bitmap[i >> 3] >> (i & 7)) & 1;
}
}  // namespace faiss

namespace cuvs::util {

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

  void run_range()
  {
    // take random imin and imax, check all ids
    std::srand(std::time(0));
    auto imin = std::rand() % spec.bitset_len;
    auto imax = std::rand() % spec.bitset_len;
    if (imin > imax) std::swap(imin, imax);
    auto selector = faiss::IDSelectorRange(imin, imax);
    auto bitset   = cuvs::core::bitset<bitset_t, index_t>(res, spec.bitset_len, false);
    auto nbits    = sizeof(bitset_t) * 8;

    cuvs::util::convert_to_bitset(res, selector, bitset.view());
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
                    << " bitset_len: " << spec.bitset_len << " imin: " << imin << " imax: " << imax
                    << " bit_element: " << bitset_converted_cpu(i / nbits));
      }
    }
  }

  void run_complex()
  {
    // generate random selectors
    std::random_device rd;
    auto seed = rd();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(1, spec.bitset_len - 1);
    auto imin = distrib(gen);
    auto imax = distrib(gen);
    if (imin > imax) std::swap(imin, imax);
    auto range_selector = faiss::IDSelectorRange(imin, imax);

    std::vector<faiss::idx_t> array_selector_indices(30);
    for (int i = 0; i < 30; i++) {
      array_selector_indices[i] = distrib(gen);
    }
    auto array_selector = faiss::IDSelectorArray(30, array_selector_indices.data());

    auto or_selector = faiss::IDSelectorOr(&range_selector, &array_selector);

    auto bitmap_faiss_cpu = std::vector<uint8_t>((spec.bitset_len + 8) / 8);
    for (uint32_t i = 0; i < bitmap_faiss_cpu.size(); i++) {
      bitmap_faiss_cpu[i] = (uint8_t)distrib(gen);
    }
    auto bitmap_selector     = faiss::IDSelectorBitmap(spec.bitset_len, bitmap_faiss_cpu.data());
    auto not_bitmap_selector = faiss::IDSelectorNot(&bitmap_selector);

    auto xor_selector = faiss::IDSelectorXOr(&or_selector, &not_bitmap_selector);

    // convert to cuVS bitset
    auto bitset = cuvs::core::bitset<bitset_t, index_t>(res, spec.bitset_len, false);
    cuvs::util::convert_to_bitset(res, xor_selector, bitset.view());

    // verify
    auto bitset_converted_cpu      = raft::make_host_vector<bitset_t, index_t>(bitset.n_elements());
    auto bitset_converted_cpu_view = cuvs::core::bitset_view<bitset_t, index_t>(
      bitset_converted_cpu.data_handle(), spec.bitset_len);
    raft::copy(res, bitset_converted_cpu.view(), bitset.to_mdspan());
    raft::resource::sync_stream(res);
    for (uint32_t i = 0; i < spec.bitset_len; i++) {
      if (bitset_converted_cpu_view.test(i) != xor_selector.is_member(i)) {
        ASSERT_TRUE(testing::AssertionFailure()
                    << "actual=" << bitset_converted_cpu_view.test(i)
                    << " != expected=" << xor_selector.is_member(i) << " @" << i
                    << " bitset_len: " << spec.bitset_len << " seed: " << seed);
      }
    }
  }

  void run_array()
  {
    // generate random array selector
    std::random_device rd;
    auto seed = rd();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(0, spec.bitset_len - 1);
    int n = spec.bitset_len / 20;  // select 5% of the bitset length
    std::vector<faiss::idx_t> array_selector_indices(n);
    for (int i = 0; i < n; i++) {
      array_selector_indices[i] = distrib(gen);
    }
    auto array_selector = faiss::IDSelectorArray(n, array_selector_indices.data());
    auto bitset         = cuvs::core::bitset<bitset_t, index_t>(res, spec.bitset_len, false);
    cuvs::util::convert_to_bitset(res, array_selector, bitset.view());

    auto bitset_converted_cpu = raft::make_host_vector<bitset_t, index_t>(bitset.n_elements());
    raft::copy(res, bitset_converted_cpu.view(), bitset.to_mdspan());
    raft::resource::sync_stream(res);
    auto bitset_converted_cpu_view = cuvs::core::bitset_view<bitset_t, index_t>(
      bitset_converted_cpu.data_handle(), spec.bitset_len);
    for (uint32_t i = 0; i < spec.bitset_len; i++) {
      if (bitset_converted_cpu_view.test(i) != array_selector.is_member(i)) {
        ASSERT_TRUE(testing::AssertionFailure()
                    << "actual=" << bitset_converted_cpu_view.test(i)
                    << " != expected=" << array_selector.is_member(i) << " @" << i
                    << " bitset_len: " << spec.bitset_len << " seed: " << seed);
      }
    }
  }
  void run()
  {
    run_range();
    run_array();
    run_complex();
  }
  raft::resources res;
  const test_spec_convert spec;
};

auto inputs_convert = ::testing::Values(test_spec_convert{100},
                                        test_spec_convert{1000},
                                        test_spec_convert{10000},
                                        test_spec_convert{200000});

using Uint32_32 = IDSelectorTest<uint32_t, uint32_t>;
TEST_P(Uint32_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(IDSelectorTest, Uint32_32, inputs_convert);
}  // namespace cuvs::util
