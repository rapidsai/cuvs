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

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * SearchResult encapsulates the logic for reading and holding search results.
 *
 * @since 25.02
 */
class BruteForceSearchResults {

   static SearchResults create(SequenceLayout neighboursSequenceLayout, SequenceLayout distancesSequenceLayout,
                               MemorySegment neighboursMemorySegment, MemorySegment distancesMemorySegment, int topK, List<Integer> mapping,
                               long numberOfQueries) {

     List<Map<Integer, Float>> results = new LinkedList<>();
     Map<Integer, Float> intermediateResultMap = new LinkedHashMap<Integer, Float>();
     var neighboursVarHandle = neighboursSequenceLayout.varHandle(MemoryLayout.PathElement.sequenceElement());
     var distancesVarHandle = distancesSequenceLayout.varHandle(MemoryLayout.PathElement.sequenceElement());

     int count = 0;
     for (long i = 0; i < topK * numberOfQueries; i++) {
       long id = (long) neighboursVarHandle.get(neighboursMemorySegment, 0L, i);
       float dst = (float) distancesVarHandle.get(distancesMemorySegment, 0L, i);
       intermediateResultMap.put(mapping != null ? mapping.get((int) id) : (int) id, dst);
       count += 1;
       if (count == topK) {
         results.add(intermediateResultMap);
         intermediateResultMap = new LinkedHashMap<>();
         count = 0;
       }
     }

     return new SearchResultsImpl(results);
  }
}
