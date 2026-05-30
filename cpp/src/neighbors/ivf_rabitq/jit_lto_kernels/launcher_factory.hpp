/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
