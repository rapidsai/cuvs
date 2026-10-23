/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../distance/top_1_nn.cuh"
#include "../neighbors/ball_cover/common.cuh"
#include "../neighbors/detail/sample_filter_data.cuh"
#include "radix_sorting_launchers.hpp"
#include <algorithm>
#include <cmath>
#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policy.cuh>
#include <cuvs/core/bitset.hpp>
#include <cuvs/core/bloom_filter.hpp>
#include <cuvs/core/cuda_fp16.hpp>
#include <limits>
#include <raft/core/bitset.cuh>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/sort.h>
#include <utility>

namespace cuvs::core {
namespace {
using default_filter_policy = cuco::bloom_filter_policy<bloom_filter::key_type>;

constexpr auto kPatternBits = default_filter_policy::pattern_bits;

constexpr auto kWordsPerBlock = default_filter_policy::words_per_block;

constexpr auto kBitsPerWord =
  std::numeric_limits<typename default_filter_policy::word_type>::digits;

static_assert(kPatternBits == kWordsPerBlock,
              "Bloom FPR sizing assumes the default policy sets one bit in each word.");

long double expected_false_positive_rate(std::size_t num_insertions, std::size_t num_blocks)
{
  if (num_insertions == 0) { return 0.0L; }

  auto const block_probability = 1.0L / static_cast<long double>(num_blocks);
  auto const bit_miss_probability =
    static_cast<long double>(kBitsPerWord - 1) / static_cast<long double>(kBitsPerWord);

  // Inclusion-exclusion evaluates the expectation over the binomial block occupancy directly:
  // E[(1 - q^X)^k] = sum_j (-1)^j C(k,j) (1 - (1 - q^j) / B)^n.
  long double result               = 0.0L;
  long double binomial_coefficient = 1.0L;
  for (std::uint32_t j = 0; j <= kPatternBits; ++j) {
    if (j > 0) {
      binomial_coefficient *=
        static_cast<long double>(kPatternBits - j + 1) / static_cast<long double>(j);
    }
    auto const avoids_selected_bits = std::pow(bit_miss_probability, j);
    auto const insertion_misses     = 1.0L - block_probability * (1.0L - avoids_selected_bits);
    auto const term = binomial_coefficient * std::pow(insertion_misses, num_insertions);
    result += (j % 2 == 0) ? term : -term;
  }

  return std::clamp(result, 0.0L, 1.0L);
}

std::size_t compute_num_blocks_from_rates(std::size_t dataset_rows,
                                          float filtering_rate,
                                          float target_false_positive_rate)
{
  RAFT_EXPECTS(dataset_rows > 0,
               "dataset_rows must be greater than zero when deriving bloom size.");
  RAFT_EXPECTS(filtering_rate > 0.0f && filtering_rate <= 1.0f,
               "filtering_rate must be in (0, 1].");
  RAFT_EXPECTS(target_false_positive_rate > 0.0f && target_false_positive_rate < 1.0f,
               "target_false_positive_rate must be in (0, 1).");

  auto expected_insertions = std::max<std::size_t>(
    1, static_cast<std::size_t>(std::ceil(static_cast<double>(dataset_rows) * filtering_rate)));
  auto const target = static_cast<long double>(target_false_positive_rate);

  std::size_t upper = 1;
  while (expected_false_positive_rate(expected_insertions, upper) > target) {
    if (upper > default_filter_policy::max_filter_blocks / 2) {
      upper = default_filter_policy::max_filter_blocks;
      RAFT_EXPECTS(expected_false_positive_rate(expected_insertions, upper) <= target,
                   "Requested Bloom filter false-positive rate requires too many blocks.");
      break;
    }
    upper *= 2;
  }

  std::size_t lower = 1;
  while (lower < upper) {
    auto const midpoint = lower + (upper - lower) / 2;
    if (expected_false_positive_rate(expected_insertions, midpoint) <= target) {
      upper = midpoint;
    } else {
      lower = midpoint + 1;
    }
  }
  return lower;
}

}  // namespace
}  // namespace cuvs::core
namespace cuvs::core {
struct bloom_filter::impl {
  using key_type              = bloom_filter::key_type;
  using cuco_filter_type      = cuco::bloom_filter<key_type>;
  using sample_filter_payload = cuvs::neighbors::detail::bloom_filter_data_t<key_type>;

  cuco_filter_type filter;
  std::size_t dataset_rows;
  float estimated_filtering_rate;

  impl(raft::resources const& res,
       std::size_t num_blocks,
       std::size_t dataset_rows_,
       float filtering_rate_)
    : filter(num_blocks, {}, {}, {}, raft::resource::get_cuda_stream(res)),
      dataset_rows(dataset_rows_),
      estimated_filtering_rate(0.0f)
  {
    auto expected_insertions = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(static_cast<double>(dataset_rows) * filtering_rate_)));
    auto expected_fpr = expected_false_positive_rate(expected_insertions, num_blocks);
    auto rejected_fraction =
      (1.0L - static_cast<long double>(filtering_rate_)) * (1.0L - expected_fpr);
    estimated_filtering_rate = static_cast<float>(std::clamp(rejected_fraction, 0.0L, 0.999L));
  }

  void validate_keys(raft::device_vector_view<const key_type, int64_t> keys) const
  {
    if (keys.extent(0) == 0) { return; }
    auto num_insertions = static_cast<std::size_t>(keys.extent(0));
    RAFT_EXPECTS(num_insertions <= dataset_rows,
                 "Number of keys to add must not exceed dataset_rows.");
  }
};

bloom_filter::bloom_filter(raft::resources const& res,
                           std::size_t dataset_rows,
                           float filtering_rate,
                           float target_false_positive_rate)
  : impl_(std::make_unique<impl>(
      res,
      compute_num_blocks_from_rates(dataset_rows, filtering_rate, target_false_positive_rate),
      dataset_rows,
      filtering_rate))
{
}

bloom_filter::~bloom_filter() = default;

bloom_filter::bloom_filter(bloom_filter&&) noexcept = default;

bloom_filter& bloom_filter::operator=(bloom_filter&&) noexcept = default;

void bloom_filter::clear(raft::resources const& res)
{
  impl_->filter.clear(raft::resource::get_cuda_stream(res));
}

void bloom_filter::clear_async(raft::resources const& res)
{
  impl_->filter.clear_async(raft::resource::get_cuda_stream(res));
}

void bloom_filter::add(raft::resources const& res,
                       raft::device_vector_view<const key_type, int64_t> keys)
{
  auto stream = raft::resource::get_cuda_stream(res);
  impl_->validate_keys(keys);
  impl_->filter.add(keys.data_handle(), keys.data_handle() + keys.extent(0), stream);
}

void bloom_filter::add_async(raft::resources const& res,
                             raft::device_vector_view<const key_type, int64_t> keys)
{
  auto stream = raft::resource::get_cuda_stream(res);
  impl_->validate_keys(keys);
  impl_->filter.add_async(keys.data_handle(), keys.data_handle() + keys.extent(0), stream);
}

void bloom_filter::contains(raft::resources const& res,
                            raft::device_vector_view<const key_type, int64_t> keys,
                            raft::device_vector_view<std::uint8_t, int64_t> output) const
{
  RAFT_EXPECTS(output.extent(0) >= keys.extent(0),
               "Bloom filter contains output size must be at least keys.size().");
  impl_->filter.contains(keys.data_handle(),
                         keys.data_handle() + keys.extent(0),
                         output.data_handle(),
                         raft::resource::get_cuda_stream(res));
}

void bloom_filter::contains_async(raft::resources const& res,
                                  raft::device_vector_view<const key_type, int64_t> keys,
                                  raft::device_vector_view<std::uint8_t, int64_t> output) const
{
  RAFT_EXPECTS(output.extent(0) >= keys.extent(0),
               "Bloom filter contains output size must be at least keys.size().");
  impl_->filter.contains_async(keys.data_handle(),
                               keys.data_handle() + keys.extent(0),
                               output.data_handle(),
                               raft::resource::get_cuda_stream(res));
}

std::size_t bloom_filter::num_blocks() const noexcept { return impl_->filter.block_extent(); }

float bloom_filter::estimate_filtering_rate() const noexcept
{
  return impl_->estimated_filtering_rate;
}

auto get_bloom_filter_impl(bloom_filter const& filter) noexcept -> bloom_filter::impl const&
{
  return *filter.impl_;
}

}  // namespace cuvs::core
namespace cuvs::neighbors::detail {
bloom_filter_data_t<std::uint32_t> bloom_filter_factory::make(
  cuvs::core::bloom_filter const& filter)
{
  return bloom_filter_data_t<std::uint32_t>{get_bloom_filter_impl(filter).filter.ref()};
}

}  // namespace cuvs::neighbors::detail
namespace cuvs::detail {
template <typename Policy, typename Iterator>
void thrust_sort(Policy const& policy, Iterator first, Iterator last)
{
  return thrust::sort(policy, first, last);
}

template <typename Policy, typename Iterator, typename Compare>
void thrust_sort(Policy const& policy, Iterator first, Iterator last, Compare compare)
{
  return thrust::sort(policy, first, last, compare);
}

template <typename Policy, typename Key, typename Value>
void thrust_sort_by_key(Policy const& policy, Key first, Key last, Value values)
{
  return thrust::sort_by_key(policy, first, last, values);
}

template <typename Policy, typename Key, typename Value, typename Compare>
void thrust_sort_by_key(Policy const& policy, Key first, Key last, Value values, Compare compare)
{
  return thrust::sort_by_key(policy, first, last, values, compare);
}

template <typename KeyIn, typename KeyOut, typename ValueIn, typename ValueOut, typename NumItems>
cudaError_t cub_sort_pairs(void* storage,
                           size_t& bytes,
                           KeyIn keys_in,
                           KeyOut keys_out,
                           ValueIn values_in,
                           ValueOut values_out,
                           NumItems count,
                           int begin_bit,
                           int end_bit,
                           cudaStream_t stream)
{
  return cub::DeviceRadixSort::SortPairs(
    storage, bytes, keys_in, keys_out, values_in, values_out, count, begin_bit, end_bit, stream);
}

template <typename InputIt, typename OutputIt, typename NumItems>
cudaError_t cub_sort_keys_descending(void* storage,
                                     size_t& bytes,
                                     InputIt input,
                                     OutputIt output,
                                     NumItems count,
                                     int begin_bit,
                                     int end_bit,
                                     cudaStream_t stream)
{
  return cub::DeviceRadixSort::SortKeysDescending(
    storage, bytes, input, output, count, begin_bit, end_bit, stream);
}

template cudaError
cuvs::detail::cub_sort_keys_descending<unsigned int*, unsigned int*, unsigned int>(
  void*, unsigned long&, unsigned int*, unsigned int*, unsigned int, int, int, CUstream_st*);

template cudaError cuvs::detail::cub_sort_pairs<int*, int*, int*, int*, int>(
  void*, unsigned long&, int*, int*, int*, int*, int, int, int, CUstream_st*);

template cudaError cuvs::detail::
  cub_sort_pairs<unsigned int const*, unsigned int*, unsigned int*, unsigned int*, unsigned long>(
    void*,
    unsigned long&,
    unsigned int const*,
    unsigned int*,
    unsigned int*,
    unsigned int*,
    unsigned long,
    int,
    int,
    CUstream_st*);

template <typename Key, typename Value>
void thrust_sort_by_key(Key first, Key last, Value values)
{
  thrust::sort_by_key(first, last, values);
}

}  // namespace cuvs::detail
template struct raft::core::bitset<uint8_t, uint32_t>;

template struct raft::core::bitset<uint16_t, uint32_t>;

template struct raft::core::bitset<uint32_t, uint32_t>;

template struct raft::core::bitset<uint32_t, int64_t>;

template struct raft::core::bitset<uint64_t, int64_t>;

template void cuvs::detail::thrust_sort(rmm::exec_policy_nosync const&,
                                        int*,
                                        int*,
                                        cuda::std::greater<int>);

template void cuvs::detail::thrust_sort(thrust::cuda_cub::par_t const&, __half*, __half*);

template void cuvs::detail::thrust_sort(thrust::cuda_cub::par_t const&, double*, double*);

template void cuvs::detail::thrust_sort(thrust::cuda_cub::par_t const&, float*, float*);

template void cuvs::detail::thrust_sort_by_key(
  rmm::exec_policy_nosync const&,
  thrust::zip_iterator<cuda::std::tuple<long*, float*>>,
  thrust::zip_iterator<cuda::std::tuple<long*, float*>>,
  long*,
  cuvs::neighbors::ball_cover::detail::NNComp);

#include "../cluster/detail/mst.cuh"

#include "../sparse/neighbors/detail/cross_component_nn.cuh"

template void cuvs::detail::thrust_sort(rmm::exec_policy_nosync const&,
                                        thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>>,
                                        thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>>,
                                        cuvs::cluster::agglomerative::detail::mst_edge_less);

template void cuvs::detail::thrust_sort(
  rmm::exec_policy_nosync const&,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>>,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>>,
  cuvs::cluster::agglomerative::detail::mst_edge_less);

template void cuvs::detail::thrust_sort_by_key(rmm::exec_policy_nosync const&, int*, int*, int*);

template void cuvs::detail::thrust_sort_by_key(rmm::exec_policy_nosync const&, long*, long*, long*);

template void cuvs::detail::thrust_sort_by_key(
  rmm::exec_policy_nosync const&,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<int*>>,
  cuvs::sparse::neighbors::detail::TupleComp);

template void cuvs::detail::thrust_sort_by_key(
  rmm::exec_policy_nosync const&,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<long*>>,
  cuvs::sparse::neighbors::detail::TupleComp);

template void cuvs::detail::thrust_sort_by_key(
  thrust::device_ptr<int>,
  thrust::device_ptr<int>,
  thrust::zip_iterator<cuda::std::tuple<thrust::device_ptr<int>, thrust::device_ptr<float>>>);

template void cuvs::detail::thrust_sort_by_key(
  thrust::device_ptr<long>,
  thrust::device_ptr<long>,
  thrust::zip_iterator<cuda::std::tuple<thrust::device_ptr<long>, thrust::device_ptr<float>>>);

#include "thrust_fill_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <raft/matrix/init.cuh>
#include <rmm/exec_policy.hpp>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/detail/tabulate.h>

namespace cuvs::detail {
template <typename Policy, typename Iterator, typename Value>
void thrust_fill(Policy const& policy, Iterator first, Iterator last, Value const& value)
{
  return thrust::fill(policy, first, last, value);
}

template <typename Policy, typename Iterator, typename Count, typename Value>
Iterator thrust_fill_n(Policy const& policy, Iterator first, Count count, Value const& value)
{
  return thrust::fill_n(policy, first, count, value);
}

template <typename Policy, typename Iterator>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last)
{
  return thrust::sequence(policy, first, last);
}

template <typename Policy, typename Iterator, typename Init>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last, Init init)
{
  return thrust::sequence(policy, first, last, init);
}

template <typename Policy, typename Iterator, typename Init, typename Step>
void thrust_sequence(Policy const& policy, Iterator first, Iterator last, Init init, Step step)
{
  return thrust::sequence(policy, first, last, init, step);
}

template <typename math_t, typename extents, typename layout>
void matrix_fill(raft::resources const& handle,
                 raft::device_mdspan<const math_t, extents, layout> in,
                 raft::device_mdspan<math_t, extents, layout> out,
                 raft::host_scalar_view<math_t> scalar)

{
  return raft::matrix::fill(handle, in, out, scalar);
}

template <typename math_t, typename extents, typename layout>
void matrix_fill(raft::resources const& handle,
                 raft::device_mdspan<math_t, extents, layout> inout,
                 math_t scalar)

{
  return raft::matrix::fill(handle, inout, scalar);
}

}  // namespace cuvs::detail
template float* cuvs::detail::thrust_fill_n(rmm::exec_policy_nosync const&,
                                            float*,
                                            unsigned long,
                                            float const&);

template unsigned int* cuvs::detail::thrust_fill_n(rmm::exec_policy_nosync const&,
                                                   unsigned int*,
                                                   unsigned long,
                                                   unsigned int const&);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  cuda::std::mdspan<
    double,
    cuda::std::extents<unsigned int, 1ul>,
    cuda::std::layout_right,
    raft::host_device_accessor<cuda::std::default_accessor<double>, (raft::memory_type)2>>,
  double);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  cuda::std::mdspan<
    float,
    cuda::std::extents<unsigned int, 1ul>,
    cuda::std::layout_right,
    raft::host_device_accessor<cuda::std::default_accessor<float>, (raft::memory_type)2>>,
  float);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_matrix_view<double, int, raft::row_major>,
                                        double);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_matrix_view<double, long, raft::row_major>,
                                        double);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_matrix_view<float, int, raft::row_major>,
                                        float);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_matrix_view<float, long, raft::row_major>,
                                        float);

template void cuvs::detail::matrix_fill(
  raft::resources const&, raft::device_matrix_view<float, unsigned long, raft::row_major>, float);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_matrix_view<int, long, raft::row_major>,
                                        int);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_matrix_view<long, long, raft::row_major>,
                                        long);

template void cuvs::detail::matrix_fill(
  raft::resources const&, raft::device_matrix_view<long, unsigned long, raft::row_major>, long);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_matrix_view<unsigned int, long, raft::row_major>,
  unsigned int);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_vector_view<bool, long, raft::row_major>,
                                        bool);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_vector_view<double, int, raft::row_major>,
                                        double);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_vector_view<double, long, raft::row_major>,
                                        double);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_vector_view<float, int, raft::row_major>,
                                        float);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_vector_view<float, long, raft::row_major>,
                                        float);

template void cuvs::detail::matrix_fill(
  raft::resources const&, raft::device_vector_view<float, unsigned long, raft::row_major>, float);

template void cuvs::detail::matrix_fill(
  raft::resources const&, raft::device_vector_view<int, unsigned long, raft::row_major>, int);

template void cuvs::detail::matrix_fill(raft::resources const&,
                                        raft::device_vector_view<long, long, raft::row_major>,
                                        long);

template void cuvs::detail::matrix_fill(
  raft::resources const&, raft::device_vector_view<long, unsigned int, raft::row_major>, long);

template void cuvs::detail::matrix_fill(
  raft::resources const&, raft::device_vector_view<long, unsigned long, raft::row_major>, long);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<raft::KeyValuePair<int, double>, int, raft::row_major>,
  raft::KeyValuePair<int, double>);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<raft::KeyValuePair<int, float>, int, raft::row_major>,
  raft::KeyValuePair<int, float>);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<raft::KeyValuePair<long, double>, long, raft::row_major>,
  raft::KeyValuePair<long, double>);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<raft::KeyValuePair<long, float>, long, raft::row_major>,
  raft::KeyValuePair<long, float>);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<unsigned int, long, raft::row_major>,
  unsigned int);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<unsigned int, unsigned int, raft::row_major>,
  unsigned int);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<unsigned int, unsigned long, raft::row_major>,
  unsigned int);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<unsigned long, int, raft::row_major>,
  unsigned long);

template void cuvs::detail::thrust_fill(rmm::exec_policy_nosync const&, int*, int*, int const&);

template void cuvs::detail::thrust_fill(thrust::cuda_cub::execute_on_stream const&,
                                        __half*,
                                        __half*,
                                        __half const&);

template void cuvs::detail::thrust_fill(thrust::cuda_cub::execute_on_stream const&,
                                        double*,
                                        double*,
                                        double const&);

template void cuvs::detail::thrust_fill(thrust::cuda_cub::execute_on_stream const&,
                                        float*,
                                        float*,
                                        float const&);

template void cuvs::detail::thrust_fill(thrust::cuda_cub::execute_on_stream const&,
                                        int*,
                                        int*,
                                        int const&);

template void cuvs::detail::matrix_fill(
  raft::resources const&,
  raft::device_vector_view<unsigned long, unsigned long, raft::row_major>,
  unsigned long);
