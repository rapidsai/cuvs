/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <cuvs/distance/distance.hpp>
#include <string>

// Use nested namespace syntax to allow inclusion from within parent namespace
namespace cuvs {
namespace neighbors {
namespace cagra {
namespace detail {
namespace multi_cta_search {

struct CagraMultiCtaSearchPlanner : CagraPlannerBase {
  CagraMultiCtaSearchPlanner() : CagraPlannerBase("search_multi_cta") {}
};

}  // namespace multi_cta_search
}  // namespace detail
}  // namespace cagra
}  // namespace neighbors
}  // namespace cuvs
