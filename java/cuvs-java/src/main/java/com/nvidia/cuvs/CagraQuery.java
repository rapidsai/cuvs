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

import com.nvidia.cuvs.CuVSMatrix.DataType;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Objects;
import java.util.function.LongToIntFunction;

/**
 * CagraQuery holds the search parameters plus either raw float[][] vectors
 * or a quantized {@link CuVSMatrix} for querying a CAGRA index.
 *
 * <p><strong>Thread Safety:</strong> Each CagraQuery instance should use its own
 * CuVSResources object that is not shared with other threads. Sharing CuVSResources
 * between threads can lead to memory allocation errors or JVM crashes.
 *
 * @since 25.02
 */
public class CagraQuery {

  private final CagraSearchParams cagraSearchParameters;
  private final float[][] queryVectors;
  private final CuVSMatrix quantizedQueries;
  private final LongToIntFunction mapping;
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
   * @param quantizedQueries      2D quantized query vector array
   * @param mapping               a function mapping ordinals (neighbor IDs) to custom user IDs
   * @param topK                  the top k results to return
   * @param prefilter             A single BitSet to use as filter while searching the CAGRA index
   * @param numDocs               Total number of dataset vectors; used to align the prefilter correctly
   * @param resources             CuVSResources instance to use for this query
   */
  private CagraQuery(
      CagraSearchParams cagraSearchParameters,
      float[][] queryVectors,
      CuVSMatrix quantizedQueries,
      LongToIntFunction mapping,
      int topK,
      BitSet prefilter,
      int numDocs,
      CuVSResources resources) {
    this.cagraSearchParameters = cagraSearchParameters;
    this.queryVectors = queryVectors;
    this.quantizedQueries = quantizedQueries;
    this.mapping = mapping;
    this.topK = topK;
    this.prefilter = prefilter;
    this.numDocs = numDocs;
    this.resources = resources;
  }

  /** Start building a new CagraQuery. */
  public static Builder newBuilder(CuVSResources resources) {
    return new Builder(resources);
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
  public CuVSMatrix getQuantizedQueries() {
    return quantizedQueries;
  }

  /** True if this query carries a quantized Dataset instead of float[][] */
  public boolean hasQuantizedQueries() {
    return quantizedQueries != null;
  }

  /**
   * Returns the data type of the query payload:
   * - 32 for float32 queries
   * - 8 for quantized queries
   */
  public DataType getQueryDataType() {
    return quantizedQueries != null ? quantizedQueries.dataType() : DataType.FLOAT;
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
    return "CagraQuery["
        + "params="
        + cagraSearchParameters
        + ", floatVectors="
        + (queryVectors != null ? Arrays.toString(queryVectors) : "null")
        + ", quantized="
        + (quantizedQueries != null ? ("Dataset@" + quantizedQueries.dataType() + "-bit") : "false")
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
    private CuVSQuantizer quantizer;
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
     * @param params an instance of the configured CagraSearchParams to
     *               be used for this query
     * @return an instance of this Builder
     */
    public Builder withSearchParams(CagraSearchParams params) {
      this.cagraSearchParams = params;
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
     * Specify a quantizer to automatically transform the float[][] queryVectors
     * into a quantized {@link CuVSMatrix} using the same quantizer used for training.
     */
    public Builder withQuantizer(CuVSQuantizer quantizer) {
      this.quantizer = quantizer;
      return this;
    }

    /**
     * Builds the CagraQuery. If a quantizer was provided, queryVectors is ignored
     * and a quantized Dataset is produced instead.
     */
    public CagraQuery build() throws Throwable {
      if (queryVectors == null) {
        throw new IllegalArgumentException("Query vectors must be provided");
      }

      CuVSMatrix quantized = null;
      float[][] floatsForQuery = queryVectors;

      if (quantizer != null) {
        // wrap float[][] in a CuVSMatrix and quantize
        try (CuVSMatrix tmp = CuVSMatrix.ofArray(queryVectors)) {
          if (tmp.dataType() != DataType.FLOAT) {
            throw new IllegalArgumentException(
                "Query quantization requires FLOAT input, got " + tmp.dataType());
          }
          quantized = quantizer.transform(tmp);
        }
        floatsForQuery = null;
      }

      return new CagraQuery(
          cagraSearchParams,
          floatsForQuery,
          quantized,
          mapping,
          topK,
          prefilter,
          numDocs,
          resources);
    }
  }
}
