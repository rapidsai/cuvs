/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
  // Left to the reader to make this thread safe
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
