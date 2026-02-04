/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

#include <raft/core/error.hpp>

FragmentDatabase::FragmentDatabase() {}

bool FragmentDatabase::make_cache_entry(std::string const& key)
{
  if (this->cache.count(key) == 0) {
    this->cache[key] = std::unique_ptr<FragmentEntry>{};
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
  RAFT_EXPECTS(val != db.cache.end(), "FragmentDatabase: Key not found: %s", key.c_str());
  return val->second.get();
}

void registerFatbinFragment(std::string const& algo,
                            std::string const& params,
                            unsigned char const* blob,
                            std::size_t size)
{
  auto& planner   = fragment_database();
  std::string key = algo;
  if (!params.empty()) { key += "_" + params; }
  auto entry_exists = planner.make_cache_entry(key);
  if (entry_exists) { return; }
  planner.cache[key] = std::make_unique<FatbinFragmentEntry>(key, blob, size);
}
