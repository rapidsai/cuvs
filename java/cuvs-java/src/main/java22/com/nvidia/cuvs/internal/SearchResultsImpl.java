package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.SearchResults;

import java.util.List;
import java.util.Map;

class SearchResultsImpl implements SearchResults {

  private final List<Map<Integer, Float>> results;

  SearchResultsImpl(List<Map<Integer, Float>> results) {
    this.results = results;
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
