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
package com.nvidia.cuvs.internal.common;

import com.nvidia.cuvs.SearchResults;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public abstract class SearchResultsImpl implements SearchResults {

  protected final List<Map<Integer, Float>> results;
  protected final List<Integer> mapping; // TODO: Is this performant in a user application?

  protected final MemorySegment neighboursMemorySegment;
  protected final MemorySegment distancesMemorySegment;
  protected final int topK;
  protected final long numberOfQueries;
  protected final VarHandle neighboursVarHandle;
  protected final VarHandle distancesVarHandle;

  protected SearchResultsImpl(
      SequenceLayout neighboursSequenceLayout,
      SequenceLayout distancesSequenceLayout,
      MemorySegment neighboursMemorySegment,
      MemorySegment distancesMemorySegment,
      int topK,
      List<Integer> mapping,
      long numberOfQueries) {
    this.topK = topK;
    this.numberOfQueries = numberOfQueries;
    this.neighboursMemorySegment = neighboursMemorySegment;
    this.distancesMemorySegment = distancesMemorySegment;
    this.mapping = mapping;
    results = new LinkedList<Map<Integer, Float>>();
    neighboursVarHandle = neighboursSequenceLayout.varHandle(PathElement.sequenceElement());
    distancesVarHandle = distancesSequenceLayout.varHandle(PathElement.sequenceElement());
  }

  /**
   * Reads neighbors and distances {@link MemorySegment} and loads the values
   * internally
   */
  protected abstract void readResultMemorySegments();

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
