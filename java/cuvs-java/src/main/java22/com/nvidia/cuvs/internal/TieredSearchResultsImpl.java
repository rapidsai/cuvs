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
