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
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.SearchResults;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.List;
import java.util.Map;
import java.util.function.LongToIntFunction;

class TieredSearchResultsImpl implements SearchResults {
  private final SequenceLayout neighboursSequenceLayout;
  private final SequenceLayout distancesSequenceLayout;
  private final MemorySegment neighboursMemorySegment;
  private final MemorySegment distancesMemorySegment;
  private final int topK;
  private final List<Integer> mapping;
  private final long numberOfQueries;

  TieredSearchResultsImpl(
      SequenceLayout neighboursSequenceLayout,
      SequenceLayout distancesSequenceLayout,
      MemorySegment neighboursMemorySegment,
      MemorySegment distancesMemorySegment,
      int topK,
      List<Integer> mapping,
      long numberOfQueries) {
    this.neighboursSequenceLayout = neighboursSequenceLayout;
    this.distancesSequenceLayout = distancesSequenceLayout;
    this.neighboursMemorySegment = neighboursMemorySegment;
    this.distancesMemorySegment = distancesMemorySegment;
    this.topK = topK;
    this.mapping = mapping;
    this.numberOfQueries = numberOfQueries;
  }

  @Override
  public List<Map<Integer, Float>> getResults() {
    // Use SearchResultsImpl.create to convert the memory segments to results
    LongToIntFunction mappingFunction = mapping != null ? (long id) -> mapping.get((int) id) : null;

    return SearchResultsImpl.create(
            neighboursSequenceLayout,
            distancesSequenceLayout,
            neighboursMemorySegment,
            distancesMemorySegment,
            topK,
            mappingFunction,
            numberOfQueries)
        .getResults();
  }
}
