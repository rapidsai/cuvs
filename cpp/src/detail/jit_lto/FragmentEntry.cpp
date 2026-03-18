/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

#include <utility>

bool FatbinFragmentEntry::add_to(nvJitLinkHandle& handle) const
{
  auto result = nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, get_data(), get_length(), get_key());

  check_nvjitlink_result(handle, result);
  return true;
}

NVRTCFatbinFragmentEntry::NVRTCFatbinFragmentEntry(std::string key, std::vector<uint8_t> program)
  : key(std::move(key)), program(std::move(program))
{
}

const uint8_t* NVRTCFatbinFragmentEntry::get_data() const { return program.data(); }

size_t NVRTCFatbinFragmentEntry::get_length() const { return program.size(); }

const char* NVRTCFatbinFragmentEntry::get_key() const { return key.c_str(); }
