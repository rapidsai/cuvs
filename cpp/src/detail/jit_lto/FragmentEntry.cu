/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvjitlink_checker.hpp"

#include <cstring>

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

FragmentEntry::FragmentEntry(std::string const& key) : compute_key(key) {}

FatbinFragmentEntry::FatbinFragmentEntry(std::string const& key,
                                         unsigned char const* view,
                                         std::size_t size)
  : FragmentEntry(key), data_size(size), data_view(view)
{
}

bool FatbinFragmentEntry::add_to(nvJitLinkHandle& handle) const
{
  auto result = nvJitLinkAddData(
    handle, NVJITLINK_INPUT_ANY, this->data_view, this->data_size, this->compute_key.c_str());

  check_nvjitlink_result(handle, result);
  return true;
}

NVRTCFragmentEntry::NVRTCFragmentEntry(std::string const& key,
                                       std::unique_ptr<char[]>&& program,
                                       std::size_t size)
  : FragmentEntry(key), program(std::move(program)), data_size(size)
{
}

bool NVRTCFragmentEntry::add_to(nvJitLinkHandle& handle) const
{
  auto result = nvJitLinkAddData(
    handle, NVJITLINK_INPUT_LTOIR, this->program.get(), this->data_size, this->compute_key.c_str());
  check_nvjitlink_result(handle, result);

  return true;
}
