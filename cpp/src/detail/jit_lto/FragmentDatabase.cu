/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include <cuvs/detail/jit_lto/FragmentDatabase.h>
#include <cuvs/detail/jit_lto/FragmentEntry.h>

FragmentDatabase::FragmentDatabase() {}

bool FragmentDatabase::make_cache_entry(std::string const& name, std::string const& params)
{
  if (this->cache.count(name + "_" + params) == 0) {
    this->cache[name + "_" + params] = std::unique_ptr<FragmentEntry>{};
    return false;
  }
  return true;
}

FragmentDatabase& fragment_database()
{
  static FragmentDatabase database;
  return database;
}

FragmentEntry* FragmentDatabase::get_fragment(std::string const& key)
{
  auto& db = fragment_database();
  auto val = db.cache.find(key);
  if (val == db.cache.end()) {
    std::cout << "FragmentDatabase: Key not found" << std::endl;
    return nullptr;
  }
  return val->second.get();
}

void registerFatbinFragment(std::string const& algo,
                            std::string const& params,
                            unsigned char const* blob,
                            std::size_t size)
{
  auto& planner     = fragment_database();
  auto entry_exists = planner.make_cache_entry(algo, params);
  if (entry_exists) { return; }
  planner.cache[algo + "_" + params] = std::make_unique<FatbinFragmentEntry>(params, blob, size);
}
