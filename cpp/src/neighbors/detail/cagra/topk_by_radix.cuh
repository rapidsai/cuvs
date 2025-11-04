/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "topk_for_cagra/topk_core.cuh"

namespace cuvs::neighbors::cagra::detail {
namespace single_cta_search {

struct topk_by_radix_sort_base {
  static constexpr std::uint32_t state_bit_length = 0;
  static constexpr std::uint32_t vecLen           = 2;  // TODO

  static constexpr uint32_t smem_size(uint32_t max_itopk) { return max_itopk * 2 + 2048 + 8; }
};

template <class IdxT>
struct topk_by_radix_sort : topk_by_radix_sort_base {
  RAFT_DEVICE_INLINE_FUNCTION void operator()(uint32_t max_topk,
                                              uint32_t num_sort_threads,
                                              uint32_t topk,
                                              uint32_t batch_size,
                                              uint32_t len_x,
                                              const uint32_t* _x,
                                              const IdxT* _in_vals,
                                              uint32_t* _y,
                                              IdxT* _out_vals,
                                              uint32_t* work,
                                              uint32_t* _hints,
                                              bool sort,
                                              uint32_t* _smem)
  {
    assert(max_topk / num_sort_threads <= 4);
    assert(blockDim.x >= V / 4);
    std::uint8_t* const state = reinterpret_cast<std::uint8_t*>(work);
    topk_cta_11_core<topk_by_radix_sort_base::state_bit_length,
                     topk_by_radix_sort_base::vecLen,
                     4,
                     IdxT>(max_topk,
                           num_sort_threads,
                           topk,
                           len_x,
                           _x,
                           _in_vals,
                           _y,
                           _out_vals,
                           state,
                           _hints,
                           sort,
                           _smem);
  }
};

}  // namespace single_cta_search
}  // namespace cuvs::neighbors::cagra::detail
