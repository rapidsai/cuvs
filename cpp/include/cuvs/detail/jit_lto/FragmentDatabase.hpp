/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "FragmentEntry.hpp"
#include "MakeFragmentKey.hpp"

class FragmentDatabase {
 public:
  FragmentDatabase(FragmentDatabase const&) = delete;
  FragmentDatabase(FragmentDatabase&&)      = delete;

  FragmentDatabase& operator=(FragmentDatabase&&)      = delete;
  FragmentDatabase& operator=(FragmentDatabase const&) = delete;

  FragmentEntry* get_fragment(std::string const& key);
  bool has_fragment(std::string const& key) const;

 private:
  FragmentDatabase();

  bool make_cache_entry(std::string const& key);

  friend FragmentDatabase& fragment_database();

  friend void registerFatbinFragment(std::string const& algo,
                                     std::string const& params,
                                     unsigned char const* blob,
                                     std::size_t size);

  friend void registerNVRTCFragment(std::string const& key,
                                    std::unique_ptr<char[]>&& program,
                                    std::size_t size);

  mutable std::mutex cache_mutex_;
  std::unordered_map<std::string, std::unique_ptr<FragmentEntry>> cache;
};

FragmentDatabase& fragment_database();

void registerFatbinFragment(std::string const& algo,
                            std::string const& params,
                            unsigned char const* blob,
                            std::size_t size);

void registerNVRTCFragment(std::string const& key,
                           std::unique_ptr<char[]>&& program,
                           std::size_t size);
