/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/cutile_module.hpp>

std::string TileAlgorithmPlanner::get_planner_key() const
{
  std::string key = this->entrypoint;
  for (const auto& fragment : cubin_fragments_) {
    key += fragment->get_key();
  }
  if (tileir_fragment_) { key += tileir_fragment_->get_key(); }
  return key;
}

std::shared_ptr<AlgorithmLauncher> TileAlgorithmPlanner::build()
{
  int cc_major = 0;
  int cc_minor = 0;
  if (!cuvs::detail::jit_lto::get_device_compute_capability(cc_major, cc_minor)) { return nullptr; }

  int driver_version = 0;
  if (cudaDriverGetVersion(&driver_version) != cudaSuccess) { return nullptr; }

  auto image = cuvs::detail::jit_lto::resolve_cutile_module_image(
    cc_major, cc_minor, driver_version, cubin_fragments_, tileir_fragment_.get());
  if (!image) { return nullptr; }

  return cuvs::detail::jit_lto::load_cutile_launcher(*image, this->entrypoint);
}
