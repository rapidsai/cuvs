/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <cuvs_internal/preprocessing/bbq_cpu_quantize.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

namespace cuvs::preprocessing::quantize::bbq::detail {

struct bbq_build_inputs {
  int64_t n_rows;
  int64_t dim;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
};

auto operator<<(std::ostream& os, const bbq_build_inputs& input) -> std::ostream&
{
  return os << "n_rows=" << input.n_rows << " dim=" << input.dim
            << " bits=" << get_bit_width(input.layout)
            << " layout=" << static_cast<int>(input.layout)
            << " metric=" << static_cast<int>(input.metric);
}

template <typename T>
auto to_host(raft::resources const& res, const T* device_ptr, size_t len) -> std::vector<T>
{
  std::vector<T> host(len);
  raft::copy(host.data(), device_ptr, len, raft::resource::get_cuda_stream(res).get());
  raft::resource::sync_stream(res);
  return host;
}

auto to_device(raft::resources const& res,
               const std::vector<float>& host,
               int64_t n_rows,
               int64_t dim) -> raft::device_matrix<float, int64_t>
{
  auto device = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  raft::copy(
    device.data_handle(), host.data(), host.size(), raft::resource::get_cuda_stream(res).get());
  raft::resource::sync_stream(res);
  return device;
}

/** True when environment variable @p name is set to something other than "", "0", "false", "off".
 */
auto env_flag_enabled(const char* name) -> bool
{
  const char* value = std::getenv(name);
  if (value == nullptr) { return false; }
  const std::string v{value};
  return !(v.empty() || v == "0" || v == "false" || v == "FALSE" || v == "off" || v == "OFF");
}

void expect_near_relative(float expected, float actual, float tolerance, const std::string& what)
{
  EXPECT_NEAR(expected, actual, tolerance * raft::max(1.0f, raft::abs(expected))) << what;
}

/**
 * Values on a 1/16 grid, small enough that a column sum over @p n_rows of them is exact in float.
 * Together with a power-of-two @p n_rows, that makes the shared centroid come out the same whether
 * it is summed serially on the host or by the device's tree reduction, which is what lets the two
 * quantizers be compared byte for byte: everything downstream of the centroid is per-row, and
 * there the device follows the host reference exactly. A grid this coarse also lands plenty of
 * components exactly halfway between two codes, which is where Java's round-half-up matters.
 */
auto make_exactly_summable_dataset(int64_t n_rows, int64_t dim, uint64_t seed) -> std::vector<float>
{
  std::mt19937 gen{seed};
  std::uniform_int_distribution<int> dist{-128, 127};
  std::vector<float> data(static_cast<size_t>(n_rows * dim));
  for (float& x : data) {
    x = static_cast<float>(dist(gen)) / 16.0f;
  }
  return data;
}

/**
 * Decodes one row back to a code per component. Written from the layout documentation in bbq.hpp
 * rather than by running the packing backwards, so that a row which reconstructs accurately is
 * evidence the codes are readable by an independent decoder -- a search kernel, or Lucene -- and
 * not just that they round-trip through the helper that wrote them.
 */
auto unpack_row(const uint8_t* row, int64_t dim, bbq_code_layout layout) -> std::vector<uint8_t>
{
  const uint32_t bits = get_bit_width(layout);
  std::vector<uint8_t> codes(static_cast<size_t>(dim), 0);
  switch (layout) {
    case bbq_code_layout::packed_7b:
    case bbq_code_layout::packed_8b: std::copy(row, row + dim, codes.begin()); break;
    case bbq_code_layout::packed_4b:
      // Dims 2k and 2k+1 share byte k, high nibble first. An odd dim has no byte for its last
      // component, which is why the tests below quantize an even number of dimensions.
      for (int64_t d = 0; d + 1 < dim; d += 2) {
        codes[d]     = row[d / 2] >> 4;
        codes[d + 1] = row[d / 2] & 0x0f;
      }
      break;
    default: {
      // One bit plane per code bit, least significant plane first, dimension d held in bit
      // 7 - d % 8 of the plane's (d / 8)-th byte. packed_1b is the single-plane case.
      const int64_t stripe = (dim + 7) / 8;
      for (int64_t d = 0; d < dim; ++d) {
        for (uint32_t bit = 0; bit < bits; ++bit) {
          const uint8_t plane_byte = row[bit * stripe + d / 8];
          codes[d] |= static_cast<uint8_t>(((plane_byte >> (7 - d % 8)) & 1u) << bit);
        }
      }
      break;
    }
  }
  return codes;
}

/** Every row of @p q decoded back to one code per component, row-major. */
auto unpack_codes(raft::resources const& res,
                  const quantizer<float, int64_t>& q,
                  int64_t n_rows,
                  int64_t dim) -> std::vector<uint8_t>
{
  const auto packed     = to_host(res, q.codes.data_handle(), q.codes.size());
  const auto row_length = static_cast<size_t>(q.codes.extent(1));
  std::vector<uint8_t> codes(static_cast<size_t>(n_rows * dim));
  for (int64_t i = 0; i < n_rows; ++i) {
    const auto row = unpack_row(packed.data() + i * row_length, dim, q.layout);
    std::copy(row.begin(), row.end(), codes.begin() + i * dim);
  }
  return codes;
}

/**
 * Mean squared reconstruction error of the quantizer over @p host_dataset, relative to the energy
 * of what it is asked to encode: the rows after the shared centroid is subtracted. Rebuilding a
 * component is what search does with the codes -- take the row's interval floor and step up by
 * the code -- so this exercises the codes, the intervals and the derived step together.
 */
auto relative_reconstruction_error(raft::resources const& res,
                                   const quantizer<float, int64_t>& q,
                                   const std::vector<uint8_t>& codes,
                                   const std::vector<float>& host_dataset,
                                   int64_t n_rows,
                                   int64_t dim) -> double
{
  const auto centroid = to_host(res, q.centroid.data_handle(), dim);
  const auto lower    = to_host(res, q.lower_intervals.data_handle(), n_rows);
  const auto delta    = to_host(res, q.dequant_delta.data_handle(), n_rows);

  double squared_error   = 0.0;
  double residual_energy = 0.0;
  for (int64_t i = 0; i < n_rows; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      const double residual = host_dataset[i * dim + d] - centroid[d];
      const double restored = lower[i] + delta[i] * codes[i * dim + d];
      squared_error += (residual - restored) * (residual - restored);
      residual_energy += residual * residual;
    }
  }
  return squared_error / residual_energy;
}

class BbqBuildTest : public ::testing::TestWithParam<bbq_build_inputs> {};

// The device build has to be a drop-in replacement for the host reference quantizer, so it is
// pinned against it field by field.
TEST_P(BbqBuildTest, MatchesHostReference)
{
  const auto input = GetParam();
  raft::resources res;

  const auto host_dataset = make_exactly_summable_dataset(input.n_rows, input.dim, 137ULL);
  auto dataset            = to_device(res, host_dataset, input.n_rows, input.dim);

  const auto expected = cuvs_internal::bbq::quantize(
    host_dataset.data(), input.n_rows, input.dim, input.metric, input.layout);
  const auto actual =
    build(res, params{input.layout, input.metric}, raft::make_const_mdspan(dataset.view()));

  ASSERT_EQ(actual.n_rows(), input.n_rows);
  ASSERT_EQ(actual.dim(), input.dim);
  ASSERT_EQ(actual.codes.extent(1), expected.codes.extent(1));
  EXPECT_EQ(actual.layout, expected.layout);
  EXPECT_EQ(actual.metric, expected.metric);

  const auto centroid = to_host(res, actual.centroid.data_handle(), input.dim);
  for (int64_t d = 0; d < input.dim; ++d) {
    EXPECT_EQ(expected.centroid(d), centroid[d]) << "centroid " << d;
  }
  EXPECT_FLOAT_EQ(expected.centroid_norm_sq, actual.centroid_norm_sq);

  const auto lower       = to_host(res, actual.lower_intervals.data_handle(), input.n_rows);
  const auto upper       = to_host(res, actual.upper_intervals.data_handle(), input.n_rows);
  const auto corrections = to_host(res, actual.additional_corrections.data_handle(), input.n_rows);
  const auto sums      = to_host(res, actual.quantized_component_sums.data_handle(), input.n_rows);
  const auto row_norm  = to_host(res, actual.row_norm.data_handle(), input.n_rows);
  const auto delta     = to_host(res, actual.dequant_delta.data_handle(), input.n_rows);
  const auto sum_delta = to_host(res, actual.dequant_sum_delta.data_handle(), input.n_rows);
  for (int64_t i = 0; i < input.n_rows; ++i) {
    const auto row = " row " + std::to_string(i);
    // The per-row reductions run across a warp rather than serially, so the corrections agree to
    // within rounding rather than exactly.
    expect_near_relative(expected.lower_intervals(i), lower[i], 1e-5f, "lower_interval" + row);
    expect_near_relative(expected.upper_intervals(i), upper[i], 1e-5f, "upper_interval" + row);
    expect_near_relative(
      expected.additional_corrections(i), corrections[i], 1e-5f, "additional_correction" + row);
    expect_near_relative(expected.row_norm(i), row_norm[i], 1e-5f, "row_norm" + row);
    expect_near_relative(expected.dequant_delta(i), delta[i], 1e-5f, "dequant_delta" + row);
    expect_near_relative(
      expected.dequant_sum_delta(i), sum_delta[i], 1e-5f, "dequant_sum_delta" + row);
    EXPECT_EQ(expected.quantized_component_sums(i), sums[i]) << "quantized_component_sum" << row;
  }

  const auto codes   = to_host(res, actual.codes.data_handle(), actual.codes.size());
  int64_t mismatches = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    if (codes[i] != expected.codes.data_handle()[i]) { ++mismatches; }
  }
  EXPECT_EQ(mismatches, 0) << mismatches << " of " << codes.size() << " code bytes differ";
}

INSTANTIATE_TEST_CASE_P(
  BbqBuildTest,
  BbqBuildTest,
  ::testing::Values(
    // Every layout, on a dimensionality that is a whole number of packing groups.
    bbq_build_inputs{512, 64, bbq_code_layout::packed_1b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{
      512, 64, bbq_code_layout::transposed_2b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{
      512, 64, bbq_code_layout::transposed_4b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{512, 64, bbq_code_layout::packed_4b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{512, 64, bbq_code_layout::packed_7b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{512, 64, bbq_code_layout::packed_8b, cuvs::distance::DistanceType::L2Expanded},
    // The correction is x.c rather than ||x - c||^2 for the non-euclidean metrics.
    bbq_build_inputs{
      512, 64, bbq_code_layout::packed_1b, cuvs::distance::DistanceType::InnerProduct},
    bbq_build_inputs{
      32 * 512, 128, bbq_code_layout::packed_4b, cuvs::distance::DistanceType::InnerProduct},
    bbq_build_inputs{
      512, 64, bbq_code_layout::packed_8b, cuvs::distance::DistanceType::CosineExpanded},
    // Rows shorter than a warp, and tails in the middle of a byte, a group and a nibble pair.
    bbq_build_inputs{64, 5, bbq_code_layout::packed_1b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{
      64, 12, bbq_code_layout::transposed_2b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{
      64, 33, bbq_code_layout::transposed_4b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{64, 41, bbq_code_layout::packed_4b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{64, 41, bbq_code_layout::packed_1b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{64, 7, bbq_code_layout::packed_8b, cuvs::distance::DistanceType::L2Expanded},
    // A single row, and enough rows to span several blocks.
    bbq_build_inputs{1, 128, bbq_code_layout::packed_1b, cuvs::distance::DistanceType::L2Expanded},
    bbq_build_inputs{
      4096, 96, bbq_code_layout::transposed_4b, cuvs::distance::DistanceType::L2Expanded}));

// On data whose centroid the host and the device cannot sum to the same float, the two can part
// ways: the interval search stops as soon as a step no longer lowers the loss, so a row sitting on
// that threshold can end up on either side of it. Nothing upstream of the search is affected, and
// the rows that do move are a handful in thousands, both of which this pins down.
TEST(BbqBuild, TracksHostReferenceOnNormalData)
{
  constexpr int64_t kRows = 5000;
  constexpr int64_t kDim  = 96;
  constexpr auto kLayout  = bbq_code_layout::transposed_4b;
  constexpr auto kMetric  = cuvs::distance::DistanceType::L2Expanded;
  raft::resources res;

  auto dataset = raft::make_device_matrix<float, int64_t>(res, kRows, kDim);
  raft::random::RngState rng{137ULL};
  raft::random::normal(res, rng, dataset.data_handle(), dataset.size(), 0.5f, 2.0f);
  const auto host_dataset = to_host(res, dataset.data_handle(), static_cast<size_t>(kRows * kDim));

  const auto expected =
    cuvs_internal::bbq::quantize(host_dataset.data(), kRows, kDim, kMetric, kLayout);
  const auto actual = build(res, params{kLayout, kMetric}, raft::make_const_mdspan(dataset.view()));

  const auto row_norm    = to_host(res, actual.row_norm.data_handle(), kRows);
  const auto corrections = to_host(res, actual.additional_corrections.data_handle(), kRows);
  const auto lower       = to_host(res, actual.lower_intervals.data_handle(), kRows);
  const auto upper       = to_host(res, actual.upper_intervals.data_handle(), kRows);
  int64_t moved_rows     = 0;
  for (int64_t i = 0; i < kRows; ++i) {
    const auto row = " row " + std::to_string(i);
    expect_near_relative(expected.row_norm(i), row_norm[i], 1e-5f, "row_norm" + row);
    expect_near_relative(
      expected.additional_corrections(i), corrections[i], 1e-5f, "additional_correction" + row);
    if (raft::abs(expected.lower_intervals(i) - lower[i]) > 1e-3f ||
        raft::abs(expected.upper_intervals(i) - upper[i]) > 1e-3f) {
      ++moved_rows;
    }
  }
  EXPECT_LE(moved_rows, kRows / 100) << moved_rows << " of " << kRows << " intervals moved";

  const auto codes   = to_host(res, actual.codes.data_handle(), actual.codes.size());
  int64_t mismatches = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    if (codes[i] != expected.codes.data_handle()[i]) { ++mismatches; }
  }
  EXPECT_LE(mismatches, static_cast<int64_t>(codes.size()) / 1000)
    << mismatches << " of " << codes.size() << " code bytes differ";
}

// What the codes are ultimately for: a row has to come back out of them. Agreeing with the host
// reference cannot show this -- the two could be wrong together -- so the codes are decoded here
// by a reader written from the layout documentation and held to an error budget per bit width.
TEST(BbqBuild, ReconstructsRowsWithinErrorBudget)
{
  constexpr int64_t kRows = 2048;
  constexpr int64_t kDim  = 128;
  constexpr auto kMetric  = cuvs::distance::DistanceType::L2Expanded;
  // Budgets are the measured error on this dataset with room to spare. Each bit buys close to the
  // 4x a uniform quantizer is worth against a gaussian, and how much of that survives is the
  // interval search's job -- these numbers are what regresses if the search stops converging.
  const std::vector<std::pair<bbq_code_layout, double>> budgets{
    {bbq_code_layout::packed_1b, 0.496},      // 0.55     measured 0.494
    {bbq_code_layout::transposed_2b, 0.127},  // 2b, 0.15     measured 0.127
    {bbq_code_layout::packed_4b, 0.01},       // 0.012     measured 0.00992
    {bbq_code_layout::transposed_4b,
     0.01},  // 4b, 0.012     same codes as packed_4b, in a different order
    {bbq_code_layout::packed_7b, 2.0e-4},   // 2.0e-4     measured 1.44e-4
    {bbq_code_layout::packed_8b, 4.0e-5}};  // 5.0e-5     measured 3.52e-5

  raft::resources res;
  auto dataset = raft::make_device_matrix<float, int64_t>(res, kRows, kDim);
  raft::random::RngState rng{137ULL};
  raft::random::normal(res, rng, dataset.data_handle(), dataset.size(), 0.5f, 2.0f);
  const auto host_dataset = to_host(res, dataset.data_handle(), static_cast<size_t>(kRows * kDim));

  std::vector<double> errors;
  std::vector<std::vector<uint8_t>> codes;
  for (const auto& [layout, budget] : budgets) {
    const auto q = build(res, params{layout, kMetric}, raft::make_const_mdspan(dataset.view()));
    codes.push_back(unpack_codes(res, q, kRows, kDim));
    errors.push_back(
      relative_reconstruction_error(res, q, codes.back(), host_dataset, kRows, kDim));
    std::cout << "[ RECON    ] " << get_bit_width(layout) << " bits, layout "
              << static_cast<int>(layout) << ": " << errors.back() << std::endl;
    EXPECT_LT(errors.back(), budget)
      << get_bit_width(layout) << " bits, layout " << static_cast<int>(layout);
  }

  // The two 4-bit layouts differ only in which byte a code lands in, so they have to carry the
  // same codes; a decoder that disagrees has misread one of the two.
  int64_t four_bit_mismatches = 0;
  for (size_t i = 0; i < codes[2].size(); ++i) {
    if (codes[2][i] != codes[3][i]) { ++four_bit_mismatches; }
  }
  EXPECT_EQ(four_bit_mismatches, 0) << four_bit_mismatches << " of " << codes[2].size()
                                    << " codes differ between packed_4b and transposed_4b";
  // Wider codes have to be worth their bytes.
  EXPECT_LT(errors[1], errors[0]) << "2 bits is no better than 1";
  EXPECT_LT(errors[2], errors[1]) << "4 bits is no better than 2";
  EXPECT_LT(errors[4], errors[2]) << "7 bits is no better than 4";
  EXPECT_LT(errors[5], errors[4]) << "8 bits is no better than 7";
}

}  // namespace cuvs::preprocessing::quantize::bbq::detail
