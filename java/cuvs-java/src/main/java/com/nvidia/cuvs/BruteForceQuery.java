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
import java.util.function.LongToIntFunction;

/**
 * BruteForceQuery holds the query vectors to be used while invoking search.
 *
 * @since 25.02
 */
public class BruteForceQuery {

  private final LongToIntFunction mapping;
  private final float[][] queryVectors;
  private final BitSet[] prefilters;
  private int numDocs = -1;
  private final int topK;

  /**
   * Constructs an instance of {@link BruteForceQuery} using queryVectors,
   * mapping, and topK.
   *
   * @param queryVectors 2D float query vector array
   * @param mapping      a function mapping ordinals (neighbor IDs) to custom user IDs
   * @param topK         the top k results to return
   * @param prefilters   the prefilters data to use while searching the BRUTEFORCE
   *                     index
   * @param numDocs      Maximum of bits in each prefilter, representing number of documents in this index.
   *                     Used only when prefilter(s) is/are passed.
   */
  public BruteForceQuery(
      float[][] queryVectors,
      LongToIntFunction mapping,
      int topK,
      BitSet[] prefilters,
      int numDocs) {
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
    this.prefilters = prefilters;
    this.numDocs = numDocs;
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

  /**
   * Gets the prefilter long array
   *
   * @return an array of bitsets
   */
  public BitSet[] getPrefilters() {
    return prefilters;
  }

  /**
   * Gets the number of documents supposed to be in this index, as used for prefilters
   *
   * @return number of documents as an integer
   */
  public int getNumDocs() {
    return numDocs;
  }

  @Override
  public String toString() {
    return "BruteForceQuery [mapping="
        + mapping
        + ", queryVectors="
        + Arrays.toString(queryVectors)
        + ", prefilter="
        + Arrays.toString(prefilters)
        + ", topK="
        + topK
        + "]";
  }

  /**
   * Builder helps configure and create an instance of BruteForceQuery.
   */
  public static class Builder {

    private float[][] queryVectors;
    private BitSet[] prefilters;
    private int numDocs;
    private LongToIntFunction mapping = SearchResults.IDENTITY_MAPPING;
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
     * Sets the prefilters data for building the {@link BruteForceQuery}.
     *
     * @param prefilters array of bitsets, as many as queries, each containing as
     *        many bits as there are vectors in the index
     * @return an instance of this Builder
     */
    public Builder withPrefilters(BitSet[] prefilters, int numDocs) {
      this.prefilters = prefilters;
      this.numDocs = numDocs;
      return this;
    }

    /**
     * Builds an instance of {@link BruteForceQuery}
     *
     * @return an instance of {@link BruteForceQuery}
     */
    public BruteForceQuery build() {
      return new BruteForceQuery(queryVectors, mapping, topK, prefilters, numDocs);
    }
  }
}
