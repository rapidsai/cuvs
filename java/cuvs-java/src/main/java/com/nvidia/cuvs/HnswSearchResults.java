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

import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import com.nvidia.cuvs.common.SearchResults;

/**
 * SearchResult encapsulates the logic for reading and holding search results.
 *
 * @since 25.02
 */
public class HnswSearchResults implements SearchResults {

  private final List<Map<Integer, Float>> results;
  private final Map<Integer, Integer> mapping; // TODO: Is this performant in a user application?
  private final SequenceLayout neighboursSequenceLayout;
  private final SequenceLayout distancesSequenceLayout;
  private final MemorySegment neighboursMemorySegment;
  private final MemorySegment distancesMemorySegment;
  private final int topK;
  private final long numberOfQueries;

  protected HnswSearchResults(SequenceLayout neighboursSequenceLayout, SequenceLayout distancesSequenceLayout,
      MemorySegment neighboursMemorySegment, MemorySegment distancesMemorySegment, int topK,
      Map<Integer, Integer> mapping, long numberOfQueries) {
    super();
    this.topK = topK;
    this.numberOfQueries = numberOfQueries;
    this.neighboursSequenceLayout = neighboursSequenceLayout;
    this.distancesSequenceLayout = distancesSequenceLayout;
    this.neighboursMemorySegment = neighboursMemorySegment;
    this.distancesMemorySegment = distancesMemorySegment;
    this.mapping = mapping;
    results = new LinkedList<Map<Integer, Float>>();

    readResultMemorySegments();
  }

  /**
   * Reads neighbors and distances {@link MemorySegment} and loads the values
   * internally
   */
  private void readResultMemorySegments() {
    VarHandle neighboursVarHandle = neighboursSequenceLayout.varHandle(PathElement.sequenceElement());
    VarHandle distancesVarHandle = distancesSequenceLayout.varHandle(PathElement.sequenceElement());

    Map<Integer, Float> intermediateResultMap = new LinkedHashMap<Integer, Float>();
    int count = 0;
    for (long i = 0; i < topK * numberOfQueries; i++) {
      long id = (long) neighboursVarHandle.get(neighboursMemorySegment, 0L, i);
      float dst = (float) distancesVarHandle.get(distancesMemorySegment, 0L, i);
      intermediateResultMap.put(mapping != null ? mapping.get((int) id) : (int) id, dst); // TODO: need to avoid this
                                                                                          // casting.
      count += 1;
      if (count == topK) {
        results.add(intermediateResultMap);
        intermediateResultMap = new LinkedHashMap<Integer, Float>();
        count = 0;
      }
    }
  }

  /**
   * Gets a list results as a map of neighbor IDs to distances.
   *
   * @return a list of results for each query as a map of neighbor IDs to distance
   */
  @Override
  public List<Map<Integer, Float>> getResults() {
    return results;
  }
}
