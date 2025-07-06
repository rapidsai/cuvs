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
package com.nvidia.cuvs;

import java.util.List;
import java.util.Map;
import java.util.function.LongToIntFunction;

public interface SearchResults {

  /**
   * The default identity function mapping neighbours IDs to user-defined IDs
   */
  LongToIntFunction IDENTITY_MAPPING = l -> (int) l;

  /**
   * Creates a mapping function from a list lookup of custom user IDs
   * @param mappingAsList a positional list of custom user IDs
   * @return a function that maps the input ordinal to a custom user IDs, using the input as an index in the list
   */
  static LongToIntFunction mappingsFromList(List<Integer> mappingAsList) {
    return l -> mappingAsList.get((int) l);
  }

  /**
   * Gets a list results as a map of neighbor IDs to distances.
   *
   * @return a list of results for each query as a map of neighbor IDs to distance
   */
  List<Map<Integer, Float>> getResults();
}
