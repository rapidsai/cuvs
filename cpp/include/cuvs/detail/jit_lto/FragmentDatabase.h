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

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "FragmentEntry.h"
#include "MakeFragmentKey.h"

struct NRTCLTOFragmentCompiler;

// struct PerEntryCachedInfo {
//   std::unordered_set<std::unique_ptr<FragmentEntry>, FragmentEntryHash,
//                      FragmentEntryEqual>
//       entries;
// };

class FragmentDatabase {
 public:
  FragmentDatabase(FragmentDatabase const&) = delete;
  FragmentDatabase(FragmentDatabase&&)      = delete;

  FragmentDatabase& operator=(FragmentDatabase&&)      = delete;
  FragmentDatabase& operator=(FragmentDatabase const&) = delete;

  std::unordered_map<std::string, std::unique_ptr<FragmentEntry>> cache;

 private:
  FragmentDatabase();

  bool make_cache_entry(std::string const& name, std::string const& params);

  friend FragmentDatabase& fragment_database();

  friend void registerFatbinFragment(std::string const& algo,
                                     std::string const& params,
                                     unsigned char const* blob,
                                     std::size_t size);
};

FragmentDatabase& fragment_database();

void registerFatbinFragment(std::string const& algo,
                            std::string const& params,
                            unsigned char const* blob,
                            std::size_t size);
