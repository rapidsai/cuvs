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
import java.util.List;

/**
 * BruteForceQuery holds the query vectors to be used while invoking search.
 *
 * @since 25.02
 */
public class BruteForceQuery {

  private List<Integer> mapping;
  private float[][] queryVectors;
  private long[] prefilter;
  private int topK;

  /**
   * Constructs an instance of {@link BruteForceQuery} using queryVectors,
   * mapping, and topK.
   *
   * @param queryVectors 2D float query vector array
   * @param mapping      an instance of ID mapping
   * @param topK         the top k results to return
   * @param prefilter    the prefilter data to use while searching the BRUTEFORCE
   *                     index
   */
  public BruteForceQuery(float[][] queryVectors, List<Integer> mapping, int topK, long[] prefilter) {
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
    this.prefilter = prefilter;
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
   * @return an integer
   */
  public int getTopK() {
    return topK;
  }

  /**
   * Gets the prefilter long array
   *
   * @return a long array
   */
  public long[] getPrefilter() {
    return prefilter;
  }

  @Override
  public String toString() {
    return "BruteForceQuery [mapping=" + mapping + ", queryVectors=" + Arrays.toString(queryVectors) + ", prefilter="
        + Arrays.toString(prefilter) + ", topK=" + topK + "]";
  }

  /**
   * Builder helps configure and create an instance of BruteForceQuery.
   */
  public static class Builder {

    private float[][] queryVectors;
    private long[] prefilter;
    private List<Integer> mapping;
    private int topK = 2;

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
     * Sets the prefilter data for building the {@link BruteForceQuery}.
     *
     * @param prefilter a one-dimensional long array
     * @return an instance of this Builder
     */
    public Builder withPrefilter(long[] prefilter) {
      this.prefilter = prefilter;
      return this;
    }

    /**
     * Builds an instance of {@link BruteForceQuery}
     *
     * @return an instance of {@link BruteForceQuery}
     */
    public BruteForceQuery build() {
      return new BruteForceQuery(queryVectors, mapping, topK, prefilter);
    }
  }
}
