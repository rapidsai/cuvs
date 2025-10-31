/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.SearchResults;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.List;
import java.util.Map;
import java.util.function.LongToIntFunction;

class TieredSearchResultsImpl implements SearchResults {
  private final List<Map<Integer, Float>> results;

  private TieredSearchResultsImpl(List<Map<Integer, Float>> results) {
    this.results = results;
  }

  public static TieredSearchResultsImpl create(
      SequenceLayout neighboursSequenceLayout,
      SequenceLayout distancesSequenceLayout,
      MemorySegment neighboursMemorySegment,
      MemorySegment distancesMemorySegment,
      int topK,
      List<Integer> mapping,
      long numberOfQueries) {

    // Process the data immediately while the memory segments are still valid
    LongToIntFunction mappingFunction = mapping != null ? (long id) -> mapping.get((int) id) : null;

    List<Map<Integer, Float>> processedResults =
        SearchResultsImpl.create(
                neighboursSequenceLayout,
                distancesSequenceLayout,
                neighboursMemorySegment,
                distancesMemorySegment,
                topK,
                mappingFunction,
                numberOfQueries)
            .getResults();

    return new TieredSearchResultsImpl(processedResults);
  }

  @Override
  public List<Map<Integer, Float>> getResults() {
    return results;
  }
}
