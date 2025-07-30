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
import java.util.function.LongToIntFunction;

/**
 * HnswQuery holds the query vectors to be used while invoking search on the
 * HNSW index.
 *
 * @since 25.02
 */
public class HnswQuery {

  private final HnswSearchParams hnswSearchParams;
  private final LongToIntFunction mapping;
  private final float[][] queryVectors;
  private final int topK;

  /**
   * Constructs an instance of {@link HnswQuery} using queryVectors, mapping, and
   * topK.
   *
   * @param hnswSearchParams the search parameters to use
   * @param queryVectors     2D float query vector array
   * @param mapping          a function mapping ordinals (neighbor IDs) to custom user IDs
   * @param topK             the top k results to return
   */
  private HnswQuery(
      HnswSearchParams hnswSearchParams,
      float[][] queryVectors,
      LongToIntFunction mapping,
      int topK) {
    this.hnswSearchParams = hnswSearchParams;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
  }

  /**
   * Gets the instance of HnswSearchParams.
   *
   * @return the instance of {@link HnswSearchParams}
   */
  public HnswSearchParams getHnswSearchParams() {
    return hnswSearchParams;
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
   * @return an integer
   */
  public int getTopK() {
    return topK;
  }

  @Override
  public String toString() {
    return "HnswQuery [mapping="
        + mapping
        + ", queryVectors="
        + Arrays.toString(queryVectors)
        + ", topK="
        + topK
        + "]";
  }

  /**
   * Builder helps configure and create an instance of BruteForceQuery.
   */
  public static class Builder {

    private HnswSearchParams hnswSearchParams;
    private float[][] queryVectors;
    private LongToIntFunction mapping = SearchResults.IDENTITY_MAPPING;
    private int topK = 2;

    /**
     * Sets the instance of configured HnswSearchParams to be passed for search.
     *
     * @param hnswSearchParams an instance of the configured HnswSearchParams to be
     *                         used for this query
     * @return an instance of this Builder
     */
    public Builder withSearchParams(HnswSearchParams hnswSearchParams) {
      this.hnswSearchParams = hnswSearchParams;
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
     * Builds an instance of {@link HnswQuery}
     *
     * @return an instance of {@link HnswQuery}
     */
    public HnswQuery build() {
      return new HnswQuery(hnswSearchParams, queryVectors, mapping, topK);
    }
  }
}
