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
 * CagraQuery holds the search parameters plus either raw float[][] vectors
 * or a quantized {@link Dataset} for querying a CAGRA index.
 *
 * @since 25.02
 */
public class CagraQuery {

  private final CagraSearchParams cagraSearchParameters;
  private final float[][] queryVectors;
  private final Dataset quantizedQueries;
  private final LongToIntFunction mapping;
  private final int topK;
  private final BitSet prefilter;
  private final int numDocs;

  private CagraQuery(
      CagraSearchParams cagraSearchParameters,
      float[][] queryVectors,
      Dataset quantizedQueries,
      LongToIntFunction mapping,
      int topK,
      BitSet prefilter,
      int numDocs) {
    this.cagraSearchParameters = cagraSearchParameters;
    this.queryVectors = queryVectors;
    this.quantizedQueries = quantizedQueries;
    this.mapping = mapping;
    this.topK = topK;
    this.prefilter = prefilter;
    this.numDocs = numDocs;
  }

  /** Start building a new CagraQuery. */
  public static Builder newBuilder() {
    return new Builder();
  }

  public CagraSearchParams getCagraSearchParameters() {
    return cagraSearchParameters;
  }

  /**
   * If this query was built without a quantizer, returns the original float vectors.
   * Otherwise returns null.
   */
  public float[][] getQueryVectors() {
    return queryVectors;
  }

  /**
   * If this query was built with a quantizer, returns the quantized Dataset.
   * Otherwise returns null.
   */
  public Dataset getQuantizedQueries() {
    return quantizedQueries;
  }

  /** True if this query carries a quantized Dataset instead of float[][] */
  public boolean hasQuantizedQueries() {
    return quantizedQueries != null;
  }

  /**
   * Returns the bit‐precision of the query payload:
   * 32 for raw floats, 8 or 1 for quantized.
   */
  public int getQueryPrecision() {
    return quantizedQueries != null ? quantizedQueries.precision() : 32;
  }

  public CuVSQuantizer getQuantizer() {
    return null;
  }

  public LongToIntFunction getMapping() {
    return mapping;
  }

  public int getTopK() {
    return topK;
  }

  public BitSet getPrefilter() {
    return prefilter;
  }

  public int getNumDocs() {
    return numDocs;
  }

  @Override
  public String toString() {
    return "CagraQuery["
        + "params="
        + cagraSearchParameters
        + ", floatVectors="
        + (queryVectors != null ? Arrays.toString(queryVectors) : "null")
        + ", quantized="
        + (quantizedQueries != null
            ? ("Dataset@" + quantizedQueries.precision() + "-bit")
            : "false")
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
    private CuVSQuantizer quantizer;

    public Builder() {}

    public Builder withSearchParams(CagraSearchParams params) {
      this.cagraSearchParams = params;
      return this;
    }

    public Builder withQueryVectors(float[][] queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    public Builder withMapping(LongToIntFunction mapping) {
      this.mapping = mapping;
      return this;
    }

    public Builder withTopK(int topK) {
      this.topK = topK;
      return this;
    }

    public Builder withPrefilter(BitSet prefilter, int numDocs) {
      this.prefilter = prefilter;
      this.numDocs = numDocs;
      return this;
    }

    /**
     * Specify a quantizer to automatically transform the float[][] queryVectors
     * into a quantized {@link Dataset} using the same quantizer used for training.
     */
    public Builder withQuantizer(CuVSQuantizer quantizer) {
      this.quantizer = quantizer;
      return this;
    }

    @Override
    public CuVSQuantizer getQuantizer() {
      return quantizer;
    }

    /**
     * Builds the CagraQuery. If a quantizer was provided, queryVectors is ignored
     * and a quantized Dataset is produced instead.
     */
    public CagraQuery build() throws Throwable {
      if (queryVectors == null) {
        throw new IllegalArgumentException("Query vectors must be provided");
      }

      Dataset quantized = null;
      float[][] floatsForQuery = queryVectors;

      if (quantizer != null) {
        // wrap float[][] in a Dataset and quantize
        Dataset tmp = Dataset.ofArray(queryVectors);
        if (tmp.precision() != 32) {
          tmp.close();
          throw new IllegalArgumentException(
              "Query quantization requires 32-bit float input, got " + tmp.precision() + "-bit");
        }
        quantized = quantizer.transform(tmp);
        tmp.close();
        floatsForQuery = null;
      }

      return new CagraQuery(
          cagraSearchParams, floatsForQuery, quantized, mapping, topK, prefilter, numDocs);
    }
  }
}
