package com.nvidia.cuvs.cagra;

import java.util.Arrays;
import java.util.Map;

import com.nvidia.cuvs.cagra.CuVSIndex.ANNAlgorithms;

public class CuVSQuery {

  SearchParams searchParams;
  PreFilter preFilter;
  float[][] queryVectors;
  public Map<Integer, Integer> mapping;
  ANNAlgorithms algo;
  
  public CuVSQuery(SearchParams searchParams, PreFilter preFilter, float[][] queryVectors,
      Map<Integer, Integer> mapping, ANNAlgorithms algo) {
    super();
    this.searchParams = searchParams;
    this.preFilter = preFilter;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.algo = algo;
  }

  @Override
  public String toString() {
    return "CuVSQuery [searchParams=" + searchParams + ", preFilter=" + preFilter + ", queries="
        + Arrays.toString(queryVectors) + "]";
  }

  public SearchParams getSearchParams() {
    return searchParams;
  }

  public PreFilter getPreFilter() {
    return preFilter;
  }

  public float[][] getQueries() {
    return queryVectors;
  }

  public static class Builder {
    SearchParams searchParams;
    PreFilter preFilter;
    float[][] queryVectors;
    Map<Integer, Integer> mapping;
    ANNAlgorithms algo = ANNAlgorithms.CAGRA;

    /**
     * 
     * @param res
     */
    public Builder() {
    }

    /**
     * 
     * @param dataset
     * @return
     */
    public Builder withSearchParams(SearchParams searchParams) {
      this.searchParams = searchParams;
      return this;
    }

    /**
     * 
     * @param queryVectors
     * @return
     */
    public Builder withQueryVectors(float[][] queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    /**
     * 
     * @param preFilter
     * @return
     */
    public Builder withPreFilter(PreFilter preFilter) {
      this.preFilter = preFilter;
      return this;
    }

    /**
     * 
     * @param mapping
     * @return
     */
    public Builder withMapping(Map<Integer, Integer> mapping) {
      this.mapping = mapping;
      return this;
    }

    /**
     * 
     * @param params
     * @return
     */
    public Builder withANNAlgorithm(ANNAlgorithms algo) {
      this.algo = algo;
      return this;
    }

    /**
     * 
     * @return
     * @throws Throwable
     */
    public CuVSQuery build() throws Throwable {
      return new CuVSQuery(searchParams, preFilter, queryVectors, mapping, algo);
    }
  }

}
