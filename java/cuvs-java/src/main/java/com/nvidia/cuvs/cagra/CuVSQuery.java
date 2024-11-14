package com.nvidia.cuvs.cagra;

import java.util.Arrays;
import java.util.Map;

public class CuVSQuery {

  private CagraSearchParams searchParams;
  private PreFilter preFilter;
  private float[][] queryVectors;
  private Map<Integer, Integer> mapping;
  private int topK;

  public CuVSQuery(CagraSearchParams searchParams, PreFilter preFilter, float[][] queryVectors,
      Map<Integer, Integer> mapping, int topK) {
    super();
    this.searchParams = searchParams;
    this.preFilter = preFilter;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
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

  public Map<Integer, Integer> getMapping() {
    return mapping;
  }

  public int getTopK() {
    return topK;
  }

  public static class Builder {
    CagraSearchParams searchParams;
    PreFilter preFilter;
    float[][] queryVectors;
    Map<Integer, Integer> mapping;
    int topK = 2;

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
     * @param topK
     * @return
     */
    public Builder withTopK(int topK) {
      this.topK = topK;
      return this;
    }

    /**
     * 
     * @return
     * @throws Throwable
     */
    public CuVSQuery build() throws Throwable {
      return new CuVSQuery(searchParams, preFilter, queryVectors, mapping, topK);
    }
  }

}
