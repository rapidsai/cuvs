/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

// Keep the public header-defined symbols that the control library exported.
// Device instantiation ownership must not determine the availability of these
// existing host ABI entry points. This file includes no kernel implementations.
template cuvs::neighbors::nn_descent::index<uint32_t>::index(
  raft::resources const&, int64_t, int64_t, bool, cuvs::distance::DistanceType);

#define CUVS_RETAIN_VAMANA_HOST_ABI(T)                                                 \
  template cuvs::neighbors::vamana::index<T, uint32_t>::index(                         \
    raft::resources const&,                                                            \
    cuvs::distance::DistanceType,                                                      \
    raft::device_matrix_view<const T, int64_t, raft::row_major>,                       \
    raft::host_matrix_view<const uint32_t, int64_t, raft::row_major>,                  \
    uint32_t);                                                                         \
  template void cuvs::neighbors::vamana::index<T, uint32_t>::update_quantized_dataset( \
    raft::resources const&, raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>);

CUVS_RETAIN_VAMANA_HOST_ABI(float)
CUVS_RETAIN_VAMANA_HOST_ABI(int8_t)
CUVS_RETAIN_VAMANA_HOST_ABI(uint8_t)
#undef CUVS_RETAIN_VAMANA_HOST_ABI

namespace cuvs::preprocessing::quantize::bbq {
CUVS_EXPORT constexpr auto get_encoded_row_length(uint32_t, bbq_code_layout) noexcept -> uint32_t;
}

namespace {
[[gnu::used]] constexpr auto keep_encoded_row_length =
  &cuvs::preprocessing::quantize::bbq::get_encoded_row_length;
}
