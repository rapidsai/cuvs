/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.SearchResults;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.function.LongToIntFunction;

/**
 * SearchResult encapsulates the logic for reading and holding search results.
 *
 * @since 25.02
 */
class BruteForceSearchResults {

  static SearchResults create(
      SequenceLayout neighboursSequenceLayout,
      SequenceLayout distancesSequenceLayout,
      MemorySegment neighboursMemorySegment,
      MemorySegment distancesMemorySegment,
      int topK,
      LongToIntFunction mapping,
      long numberOfQueries) {
    return SearchResultsImpl.create(
        neighboursSequenceLayout,
        distancesSequenceLayout,
        neighboursMemorySegment,
        distancesMemorySegment,
        topK,
        mapping,
        numberOfQueries);
  }
}
