/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

package com.nvidia.cuvs.cagra;

import java.util.Arrays;
import java.util.Map;

public class CuVSQuery {

  private CagraSearchParams cagraSearchParameters;
  private PreFilter preFilter;
  private float[][] queryVectors;
  private Map<Integer, Integer> mapping;
  private int topK;

  public CuVSQuery(CagraSearchParams cagraSearchParameters, PreFilter preFilter, float[][] queryVectors,
      Map<Integer, Integer> mapping, int topK) {
    super();
    this.cagraSearchParameters = cagraSearchParameters;
    this.preFilter = preFilter;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
  }

  public CagraSearchParams getCagraSearchParameters() {
    return cagraSearchParameters;
  }

  public PreFilter getPreFilter() {
    return preFilter;
  }

  public float[][] getQueryVectors() {
    return queryVectors;
  }

  public Map<Integer, Integer> getMapping() {
    return mapping;
  }

  public int getTopK() {
    return topK;
  }

  @Override
  public String toString() {
    return "CuVSQuery [cagraSearchParameters=" + cagraSearchParameters + ", preFilter=" + preFilter + ", queryVectors="
        + Arrays.toString(queryVectors) + ", mapping=" + mapping + ", topK=" + topK + "]";
  }

  public static class Builder {

    private CagraSearchParams cagraSearchParams;
    private PreFilter preFilter;
    private float[][] queryVectors;
    private Map<Integer, Integer> mapping;
    private int topK = 2;

    public Builder withSearchParams(CagraSearchParams cagraSearchParams) {
      this.cagraSearchParams = cagraSearchParams;
      return this;
    }

    public Builder withQueryVectors(float[][] queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    public Builder withPreFilter(PreFilter preFilter) {
      this.preFilter = preFilter;
      return this;
    }

    public Builder withMapping(Map<Integer, Integer> mapping) {
      this.mapping = mapping;
      return this;
    }

    public Builder withTopK(int topK) {
      this.topK = topK;
      return this;
    }

    public CuVSQuery build() throws Throwable {
      return new CuVSQuery(cagraSearchParams, preFilter, queryVectors, mapping, topK);
    }
  }
}
