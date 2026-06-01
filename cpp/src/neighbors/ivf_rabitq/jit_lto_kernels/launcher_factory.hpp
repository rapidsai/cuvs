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
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut_block_sort_launcher(
  bool with_ex)
{
  ComputeInnerProductsWithLutBlockSortPlanner planner;
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut16_opt_launcher(
  bool with_ex)
{
  ComputeInnerProductsWithLut16OptPlanner planner;
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher>
make_compute_inner_products_with_lut16_opt_block_sort_launcher(bool with_ex)
{
  ComputeInnerProductsWithLut16OptBlockSortPlanner planner;
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_bitwise_launcher(
  bool with_ex)
{
  ComputeInnerProductsWithBitwisePlanner planner;
  if (with_ex) {
    planner.add_entrypoint<true>();
    planner.add_extract_code_device_function();
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher>
make_compute_inner_products_with_bitwise_block_sort_launcher(int num_bits, bool with_ex)
{
  ComputeInnerProductsWithBitwiseBlockSortPlanner planner;
  if (num_bits == 4) {
    if (with_ex) {
      planner.add_entrypoint<4, true>();
      planner.add_extract_code_device_function();
    } else {
      planner.add_entrypoint<4, false>();
    }
  } else {
    if (with_ex) {
      planner.add_entrypoint<8, true>();
      planner.add_extract_code_device_function();
    } else {
      planner.add_entrypoint<8, false>();
    }
  }
  return planner.get_launcher();
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
