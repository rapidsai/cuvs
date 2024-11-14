package com.nvidia.cuvs.cagra;

import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class SearchResult {

  private List<Map<Integer, Float>> results;
  private Map<Integer, Integer> mapping;
  private SequenceLayout neighboursSL;
  private SequenceLayout distancesSL;
  private MemorySegment neighboursMS;
  private MemorySegment distancesMS;
  private int topK;
  private int numQueries;

  public SearchResult(SequenceLayout neighboursSL, SequenceLayout distancesSL, MemorySegment neighboursMS,
      MemorySegment distancesMS, int topK, Map<Integer, Integer> mapping, int numQueries) {
    super();
    this.topK = topK;
    this.numQueries = numQueries;
    this.neighboursSL = neighboursSL;
    this.distancesSL = distancesSL;
    this.neighboursMS = neighboursMS;
    this.distancesMS = distancesMS;
    this.mapping = mapping;
    results = new LinkedList<Map<Integer, Float>>();
    this.load();
  }

  private void load() {
    VarHandle neighboursVH = neighboursSL.varHandle(PathElement.sequenceElement());
    VarHandle distancesVH = distancesSL.varHandle(PathElement.sequenceElement());

    Map<Integer, Float> irm = new LinkedHashMap<Integer, Float>();
    int count = 0;
    for (long i = 0; i < topK * numQueries; i++) {
      int id = (int) neighboursVH.get(neighboursMS, 0L, i);
      float dst = (float) distancesVH.get(distancesMS, 0L, i);
      irm.put(mapping != null ? mapping.get(id) : id, dst);
      count += 1;
      if (count == topK) {
        results.add(irm);
        irm = new LinkedHashMap<Integer, Float>();
        count = 0;
      }
    }
  }

  public List<Map<Integer, Float>> getResults() {
    return results;
  }

}
