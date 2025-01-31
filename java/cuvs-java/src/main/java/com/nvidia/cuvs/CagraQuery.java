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
 * CagraQuery holds the CagraSearchParams and the query vectors to be used while
 * invoking search.
 *
 * @since 25.02
 */
public class CagraQuery {

  private CagraSearchParams cagraSearchParameters;
  private List<Integer> mapping;
  private float[][] queryVectors;
  private int topK;

  /**
   * Constructs an instance of {@link CagraQuery} using cagraSearchParameters,
   * preFilter, queryVectors, mapping, and topK.
   *
   * @param cagraSearchParameters an instance of {@link CagraSearchParams} holding
   *                              the search parameters
   * @param queryVectors          2D float query vector array
   * @param mapping               an instance of ID mapping
   * @param topK                  the top k results to return
   */
  public CagraQuery(CagraSearchParams cagraSearchParameters, float[][] queryVectors, List<Integer> mapping, int topK) {
    super();
    this.cagraSearchParameters = cagraSearchParameters;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
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

  @Override
  public String toString() {
    return "CuVSQuery [cagraSearchParameters=" + cagraSearchParameters + ", queryVectors="
        + Arrays.toString(queryVectors) + ", mapping=" + mapping + ", topK=" + topK + "]";
  }

  /**
   * Builder helps configure and create an instance of CagraQuery.
   */
  public static class Builder {

    private CagraSearchParams cagraSearchParams;
    private float[][] queryVectors;
    private List<Integer> mapping;
    private int topK = 2;

    /**
     * Default constructor.
     */
    public Builder() {
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
     * Builds an instance of CuVSQuery.
     *
     * @return an instance of CuVSQuery
     */
    public CagraQuery build() {
      return new CagraQuery(cagraSearchParams, queryVectors, mapping, topK);
    }
  }
}
