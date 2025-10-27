/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
