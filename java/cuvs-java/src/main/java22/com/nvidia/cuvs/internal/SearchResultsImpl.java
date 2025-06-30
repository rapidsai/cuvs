package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.SearchResults;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

class SearchResultsImpl implements SearchResults {

  private final List<Map<Integer, Float>> results;

  SearchResultsImpl(List<Map<Integer, Float>> results) {
    this.results = results;
  }

  /**
   * Factory method to create an on-heap SearchResults (backed by standard Java data types and containers) from
   * native/off-heap memory data structures.
   */
  static SearchResults create(SequenceLayout neighboursSequenceLayout, SequenceLayout distancesSequenceLayout,
                              MemorySegment neighboursMemorySegment, MemorySegment distancesMemorySegment, int topK, List<Integer> mapping,
                              long numberOfQueries) {
    List<Map<Integer, Float>> results = new LinkedList<>();
    Map<Integer, Float> intermediateResultMap = new LinkedHashMap<>();
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
