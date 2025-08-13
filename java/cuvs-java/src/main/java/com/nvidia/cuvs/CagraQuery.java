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

import java.util.Arrays;
import java.util.BitSet;
import java.util.Objects;
import java.util.function.LongToIntFunction;

/**
 * CagraQuery holds the CagraSearchParams and the query vectors to be used while
 * invoking search.
 *
 * <p><strong>Thread Safety:</strong> Each CagraQuery instance should use its own
 * CuVSResources object that is not shared with other threads. Sharing CuVSResources
 * between threads can lead to memory allocation errors or JVM crashes.
 *
 * @since 25.02
 */
public class CagraQuery {

  private final CagraSearchParams cagraSearchParameters;
  private final LongToIntFunction mapping;
  private final float[][] queryVectors;
  private final int topK;
  private final BitSet prefilter;
  private final int numDocs;
  private final CuVSResources resources;

  /**
   * Constructs an instance of {@link CagraQuery} using cagraSearchParameters,
   * preFilter, queryVectors, mapping, and topK.
   *
   * @param cagraSearchParameters an instance of {@link CagraSearchParams} holding
   *                              the search parameters
   * @param queryVectors          2D float query vector array
   * @param mapping               a function mapping ordinals (neighbor IDs) to custom user IDs
   * @param topK                  the top k results to return
   * @param prefilter             A single BitSet to use as filter while searching the CAGRA index
   * @param numDocs               Total number of dataset vectors; used to align the prefilter correctly
   * @param resources             CuVSResources instance to use for this query
   */
  public CagraQuery(
      CagraSearchParams cagraSearchParameters,
      float[][] queryVectors,
      LongToIntFunction mapping,
      int topK,
      BitSet prefilter,
      int numDocs,
      CuVSResources resources) {
    super();
    this.cagraSearchParameters = cagraSearchParameters;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
    this.prefilter = prefilter;
    this.numDocs = numDocs;
    this.resources = resources;
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
   * Gets the function mapping ordinals (neighbor IDs) to custom user IDs
   */
  public LongToIntFunction getMapping() {
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
   * Gets the number of documents in this index, as used for prefilter
   *
   * @return number of documents as an integer
   */
  public int getNumDocs() {
    return numDocs;
  }

  /**
   * Gets the CuVSResources instance for this query.
   *
   * @return the CuVSResources instance
   */
  public CuVSResources getResources() {
    return resources;
  }

  @Override
  public String toString() {
    return "CuVSQuery [cagraSearchParameters="
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
   * Builder helps configure and create an instance of CagraQuery.
   */
  public static class Builder {

    private CagraSearchParams cagraSearchParams;
    private float[][] queryVectors;
    private LongToIntFunction mapping = SearchResults.IDENTITY_MAPPING;
    private int topK = 2;
    private BitSet prefilter;
    private int numDocs;
    private final CuVSResources resources;

    /**
     * Constructor that requires CuVSResources.
     *
     * <p><strong>Important:</strong> The provided CuVSResources instance should not be
     * shared with other threads. Each thread performing searches should create its own
     * CuVSResources instance to avoid memory allocation conflicts and potential JVM crashes.
     *
     * @param resources the CuVSResources instance to use for this query (must not be shared between threads)
     */
    public Builder(CuVSResources resources) {
      this.resources = Objects.requireNonNull(resources, "resources cannot be null");
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
     * Sets the function used to map ordinals (neighbor IDs) to custom user IDs
     *
     * @param mapping a function mapping ordinals (neighbor IDs) to custom user IDs
     * @return an instance of this Builder
     */
    public Builder withMapping(LongToIntFunction mapping) {
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
     * Sets a global prefilter for all queries in this {@link CagraQuery}.
     * The {@code prefilter} array must contain exactly one {@link BitSet},
     * which is applied to all queries. A bit value of {@code 1} includes the
     * corresponding dataset vector; {@code 0} excludes it.
     *
     * @param prefilter an array with the global filter BitSet
     * @param numDocs total number of vectors in the dataset (for alignment)
     * @return this {@link Builder} instance
     */
    public Builder withPrefilter(BitSet prefilter, int numDocs) {
      this.prefilter = prefilter;
      this.numDocs = numDocs;
      return this;
    }

    /**
     * Builds an instance of CuVSQuery.
     *
     * @return an instance of CuVSQuery
     */
    public CagraQuery build() {
      return new CagraQuery(
          cagraSearchParams, queryVectors, mapping, topK, prefilter, numDocs, resources);
    }
  }
}
