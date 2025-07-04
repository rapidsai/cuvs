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
package com.nvidia.cuvs;

import com.nvidia.cuvs.TieredIndex.TieredIndexType;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

/**
 * TieredIndexQuery holds the search parameters and query vectors to be used
 * while
 * invoking search. Currently only supports CAGRA index type.
 *
 * @since 25.02
 */
public class TieredIndexQuery {
  private TieredIndexType indexType;
  private CagraSearchParams cagraSearchParameters;
  private List<Integer> mapping;
  private float[][] queryVectors;
  private int topK;
  private BitSet prefilter;
  private long numDocs;

  private TieredIndexQuery(
      TieredIndexType indexType,
      CagraSearchParams cagraSearchParameters,
      List<Integer> mapping,
      float[][] queryVectors,
      int topK,
      BitSet prefilter,
      long numDocs) {
    super();
    this.indexType = indexType;
    this.cagraSearchParameters = cagraSearchParameters;
    this.mapping = mapping;
    this.queryVectors = queryVectors;
    this.topK = topK;
    this.prefilter = prefilter;
    this.numDocs = numDocs;
  }

  /**
   * Gets the index type for this query.
   *
   * @return the TieredIndexType
   */
  public TieredIndexType getIndexType() {
    return indexType;
  }

  /**
   * Gets the instance of CagraSearchParams initially set.
   *
   * @return an instance CagraSearchParams
   */
  public CagraSearchParams getCagraSearchParameters() {
    return cagraSearchParameters;
  }

  /**
   * Gets the query vector 2D float array.
   *
   * @return 2D float array
   */
  public float[][] getQueryVectors() {
    return queryVectors;
  }

  /**
   * Gets the passed map instance.
   *
   * @return a map of ID mappings
   */
  public List<Integer> getMapping() {
    return mapping;
  }

  /**
   * Gets the topK value.
   *
   * @return the topK value
   */
  public int getTopK() {
    return topK;
  }

  /**
   * Gets the prefilter BitSet.
   *
   * @return a BitSet object representing the prefilter
   */
  public BitSet getPrefilter() {
    return prefilter;
  }

  /**
   * Gets the number of documents in this index, as used for prefilter.
   *
   * @return number of documents as an integer
   */
  public long getNumDocs() {
    return numDocs;
  }

  @Override
  public String toString() {
    return "TieredIndexQuery [indexType="
        + indexType
        + ", cagraSearchParameters="
        + cagraSearchParameters
        + ", queryVectors="
        + Arrays.toString(queryVectors)
        + ", mapping="
        + mapping
        + ", topK="
        + topK
        + "]";
  }

  /**
   * Creates a new Builder instance.
   *
   * @return a new Builder instance
   */
  public static Builder newBuilder() {
    return new Builder();
  }

  /**
   * Builder helps configure and create an instance of TieredIndexQuery.
   */
  public static class Builder {
    private TieredIndexType indexType = TieredIndexType.CAGRA;
    private CagraSearchParams cagraSearchParams;
    private float[][] queryVectors;
    private List<Integer> mapping;
    private int topK = 2;
    private BitSet prefilter;
    private long numDocs;

    /**
     * Sets the index type for this query.
     *
     * @param indexType the index type
     * @return an instance of this Builder
     */
    public Builder withIndexType(TieredIndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    /**
     * Sets the instance of configured CagraSearchParams to be passed for search.
     *
     * @param cagraSearchParams an instance of the configured CagraSearchParams to
     *                          be used for this query
     * @return an instance of this Builder
     */
    public Builder withSearchParams(CagraSearchParams cagraSearchParams) {
      this.cagraSearchParams = cagraSearchParams;
      return this;
    }

    /**
     * Registers the query vectors to be passed in the search call.
     *
     * @param queryVectors 2D float query vector array
     * @return an instance of this Builder
     */
    public Builder withQueryVectors(float[][] queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    /**
     * Sets the instance of mapping to be used for ID mapping.
     *
     * @param mapping the ID mapping instance
     * @return an instance of this Builder
     */
    public Builder withMapping(List<Integer> mapping) {
      this.mapping = mapping;
      return this;
    }

    /**
     * Registers the topK value.
     *
     * @param topK the topK value used to retrieve the topK results
     * @return an instance of this Builder
     */
    public Builder withTopK(int topK) {
      this.topK = topK;
      return this;
    }

    /**
     * Sets a BitSet to use as prefilter while searching.
     *
     * @param prefilter the BitSet to use as prefilter
     * @param numDocs   Total number of dataset vectors; used to align the prefilter
     *                  correctly
     * @return an instance of this Builder
     */
    public Builder withPrefilter(BitSet prefilter, int numDocs) {
      this.prefilter = prefilter;
      this.numDocs = numDocs;
      return this;
    }

    /**
     * Builds an instance of TieredIndexQuery.
     *
     * @return an instance of TieredIndexQuery
     * @throws IllegalStateException if required parameters are missing
     */
    public TieredIndexQuery build() {
      return new TieredIndexQuery(
          indexType, cagraSearchParams, mapping, queryVectors, topK, prefilter, numDocs);
    }
  }
}
