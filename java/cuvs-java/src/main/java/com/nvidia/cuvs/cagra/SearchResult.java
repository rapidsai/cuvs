/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

package com.nvidia.cuvs.cagra;

import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * SearchResult encapsulates the logic for reading and holding search results.
 * 
 * @since 24.12
 */
public class SearchResult {

  private List<Map<Integer, Float>> results;
  private Map<Integer, Integer> mapping;
  private SequenceLayout neighboursSequenceLayout;
  private SequenceLayout distancesSequenceLayout;
  private MemorySegment neighboursMemorySegment;
  private MemorySegment distancesMemorySegment;
  private int topK;
  private int numberOfQueries;

  /**
   * Constructor that initializes SearchResult with neighboursSequenceLayout,
   * distancesSequenceLayout, neighboursMemorySegment, distancesMemorySegment,
   * topK, mapping, and numberOfQueries.
   * 
   * @param neighboursSequenceLayout neighbor SequenceLayout instance
   * @param distancesSequenceLayout  distance SequenceLayout instance
   * @param neighboursMemorySegment  neighbor MemorySegment instance
   * @param distancesMemorySegment   distance MemorySegment instance
   * @param topK                     an integer denoting the topK value
   * @param mapping                  id mapping
   * @param numberOfQueries          number of queries that were initially
   *                                 submitted
   * @see SequenceLayout
   * @see MemorySegment
   */
  public SearchResult(SequenceLayout neighboursSequenceLayout, SequenceLayout distancesSequenceLayout,
      MemorySegment neighboursMemorySegment, MemorySegment distancesMemorySegment, int topK,
      Map<Integer, Integer> mapping, int numberOfQueries) {
    super();
    this.topK = topK;
    this.numberOfQueries = numberOfQueries;
    this.neighboursSequenceLayout = neighboursSequenceLayout;
    this.distancesSequenceLayout = distancesSequenceLayout;
    this.neighboursMemorySegment = neighboursMemorySegment;
    this.distancesMemorySegment = distancesMemorySegment;
    this.mapping = mapping;
    results = new LinkedList<Map<Integer, Float>>();
    this.readResultMemorySegments();
  }

  /**
   * Reads neighbors and distances MemorySegments and load values in a List of
   * Maps.
   * 
   * @see MemorySegment
   * @see VarHandle
   */
  private void readResultMemorySegments() {
    VarHandle neighboursVarHandle = neighboursSequenceLayout.varHandle(PathElement.sequenceElement());
    VarHandle distancesVarHandle = distancesSequenceLayout.varHandle(PathElement.sequenceElement());

    Map<Integer, Float> intermediateResultMap = new LinkedHashMap<Integer, Float>();
    int count = 0;
    for (long i = 0; i < topK * numberOfQueries; i++) {
      int id = (int) neighboursVarHandle.get(neighboursMemorySegment, 0L, i);
      float dst = (float) distancesVarHandle.get(distancesMemorySegment, 0L, i);
      intermediateResultMap.put(mapping != null ? mapping.get(id) : id, dst);
      count += 1;
      if (count == topK) {
        results.add(intermediateResultMap);
        intermediateResultMap = new LinkedHashMap<Integer, Float>();
        count = 0;
      }
    }
  }

  /**
   * Gets a list of maps containing topK ID and distances.
   * 
   * @return a list of maps
   */
  public List<Map<Integer, Float>> getResults() {
    return results;
  }
}
