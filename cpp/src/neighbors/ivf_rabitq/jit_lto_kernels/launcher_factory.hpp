/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_inner_products_with_bitwise_block_sort_planner.hpp"
#include "compute_inner_products_with_bitwise_planner.hpp"
#include "compute_inner_products_with_lut16_opt_block_sort_planner.hpp"
#include "compute_inner_products_with_lut16_opt_planner.hpp"
#include "compute_inner_products_with_lut_block_sort_planner.hpp"
#include "compute_inner_products_with_lut_planner.hpp"

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>

#include <memory>

namespace cuvs::neighbors::ivf_rabitq::detail {

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut_launcher(
  bool with_ex)
{
  ComputeInnerProductsWithLutPlanner planner;
  planner.add_compute_lut_ip_for_vec_device_function();
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
    planner.add_compute_ip2_from_long_codes_warp_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut_block_sort_launcher(
  bool with_ex)
{
  ComputeInnerProductsWithLutBlockSortPlanner planner;
  planner.add_compute_lut_ip_for_vec_device_function();
  planner.add_update_threshold_atomicmin_device_function();
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
    planner.add_compute_ip2_from_long_codes_warp_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut16_opt_launcher(
  bool with_ex)
{
  ComputeInnerProductsWithLut16OptPlanner planner;
  planner.add_compute_lut_ip_for_vec_device_function();
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
    planner.add_compute_ip2_from_long_codes_warp_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher>
make_compute_inner_products_with_lut16_opt_block_sort_launcher(bool with_ex)
{
  ComputeInnerProductsWithLut16OptBlockSortPlanner planner;
  planner.add_compute_lut_ip_for_vec_device_function();
  planner.add_update_threshold_atomicmin_device_function();
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
    planner.add_compute_ip2_from_long_codes_warp_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_bitwise_launcher(
  bool with_ex)
{
  ComputeInnerProductsWithBitwisePlanner planner;
  planner.add_compute_bitwise_1bit_ip_for_vec_device_function();
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
    planner.add_compute_ip2_from_long_codes_warp_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher>
make_compute_inner_products_with_bitwise_block_sort_launcher(int num_bits, bool with_ex)
{
  ComputeInnerProductsWithBitwiseBlockSortPlanner planner;
  planner.add_compute_bitwise_1bit_ip_for_vec_device_function();
  if (num_bits == 4) {
    planner.add_compute_bitwise_quantized_ip_for_vec_device_function<4>();
  } else {
    planner.add_compute_bitwise_quantized_ip_for_vec_device_function<8>();
  }
  planner.add_update_threshold_atomicmin_device_function();
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
    planner.add_compute_ip2_from_long_codes_warp_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
