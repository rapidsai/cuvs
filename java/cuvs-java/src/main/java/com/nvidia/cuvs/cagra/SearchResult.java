package com.nvidia.cuvs.cagra;

import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.HashMap;
import java.util.Map;

public class SearchResult {

  private Map<Integer, Float> results;
  private Map<Integer, Integer> mapping;
  SequenceLayout neighboursSL;
  SequenceLayout distancesSL;
  MemorySegment neighboursMS;
  MemorySegment distancesMS;
  int topK;

  public SearchResult(SequenceLayout neighboursSL, SequenceLayout distancesSL, MemorySegment neighboursMS,
      MemorySegment distancesMS, int topK, Map<Integer, Integer> mapping) {
    super();
    this.topK = topK;
    this.neighboursSL = neighboursSL;
    this.distancesSL = distancesSL;
    this.neighboursMS = neighboursMS;
    this.distancesMS = distancesMS;
    this.mapping = mapping;
    results = new HashMap<Integer, Float>();
    this.load();
  }

  private void load() {
    VarHandle neighboursVH = neighboursSL.varHandle(PathElement.sequenceElement());
    VarHandle distancesVH = distancesSL.varHandle(PathElement.sequenceElement());

    for (long i = 0; i < topK; i++) {
      int id = (int) neighboursVH.get(neighboursMS, 0L, i);
      results.put(mapping != null ? mapping.get(id) : id, (float) distancesVH.get(distancesMS, 0L, i));
    }
  }

  public Map<Integer, Float> getResults() {
    return results;
  }

}
