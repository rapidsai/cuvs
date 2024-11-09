package com.nvidia.cuvs.cagra;

import java.util.Arrays;
import java.util.Map;

public class CuVSQuery {

  CagraSearchParams searchParams;
  PreFilter preFilter;
  float[][] queryVectors;
  public Map<Integer, Integer> mapping;

  public CuVSQuery(CagraSearchParams searchParams, PreFilter preFilter, float[][] queryVectors,
      Map<Integer, Integer> mapping) {
    super();
    this.searchParams = searchParams;
    this.preFilter = preFilter;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
  }

  @Override
  public String toString() {
    return "CuVSQuery [searchParams=" + searchParams + ", preFilter=" + preFilter + ", queries="
        + Arrays.toString(queryVectors) + "]";
  }

  public CagraSearchParams getSearchParams() {
    return searchParams;
  }

  public PreFilter getPreFilter() {
    return preFilter;
  }

  public float[][] getQueries() {
    return queryVectors;
  }

  public static class Builder {
    CagraSearchParams searchParams;
    PreFilter preFilter;
    float[][] queryVectors;
    Map<Integer, Integer> mapping;

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
    public Builder withSearchParams(CagraSearchParams searchParams) {
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
     * @return
     * @throws Throwable
     */
    public CuVSQuery build() throws Throwable {
      return new CuVSQuery(searchParams, preFilter, queryVectors, mapping);
    }
  }

}
