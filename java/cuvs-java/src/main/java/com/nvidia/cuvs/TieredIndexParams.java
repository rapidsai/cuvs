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

import java.util.Objects;

/**
 * Configuration parameters for building a {@link TieredIndex}.
 * Only CAGRA is currently supported as the underlying ANN algorithm.
 *
 */
public final class TieredIndexParams {

  /**
   * Enumeration of supported distance metrics for TieredIndex.
   */
  public enum Metric {
    /** L2 (Euclidean) distance metric */
    L2,
    /** Inner product (cosine similarity) distance metric */
    INNER_PRODUCT
  }

  private final Metric metric;
  private final int minAnnRows;
  private final boolean createAnnIndexOnExtend;
  private final CagraIndexParams cagraParams;

  /**
   * Private constructor used by the Builder.
   *
   * @param builder The Builder instance containing the configuration
   */
  private TieredIndexParams(Builder builder) {
    this.metric = builder.metric;
    this.minAnnRows = builder.minAnnRows;
    this.createAnnIndexOnExtend = builder.createAnnIndexOnExtend;
    this.cagraParams = builder.cagraParams;
  }

  /**
   * Returns the distance metric used for similarity computation.
   *
   * @return The {@link Metric} (L2 or Inner Product)
   */
  public Metric getMetric() {
    return metric;
  }

  /**
   * Returns the minimum number of rows required to use the ANN algorithm.
   *
   * @return The minimum row count threshold for ANN algorithm usage
   */
  public int getMinAnnRows() {
    return minAnnRows;
  }

  /**
   * Returns whether to create an ANN index when extending the dataset.
   *
   * @return true if ANN index should be created on extend, false otherwise
   */
  public boolean isCreateAnnIndexOnExtend() {
    return createAnnIndexOnExtend;
  }

  /**
   * Returns the CAGRA-specific parameters for the ANN algorithm.
   *
   * @return The {@link CagraIndexParams} configuration, or null if not using
   *         CAGRA
   */
  public CagraIndexParams getCagraParams() {
    return cagraParams;
  }

  /**
   * Creates a new Builder for constructing TieredIndexParams.
   *
   * @return A new Builder instance
   */
  public static Builder newBuilder() {
    return new Builder();
  }

  /**
   * Builder class for constructing {@link TieredIndexParams} instances.
   */
  public static final class Builder {
    private Metric metric = Metric.L2;
    private int minAnnRows = 4096;
    private boolean createAnnIndexOnExtend = true;
    private CagraIndexParams cagraParams = null;

    /**
     * Sets the distance metric for similarity computation.
     *
     * @param metric The {@link Metric} to use (L2 or Inner Product)
     * @return This Builder instance for method chaining
     * @throws NullPointerException if metric is null
     */
    public Builder metric(Metric metric) {
      this.metric = Objects.requireNonNull(metric);
      return this;
    }

    /**
     * Sets the minimum number of rows required to use the ANN algorithm.
     *
     * @param minAnnRows The minimum row count threshold (must be positive)
     * @return This Builder instance for method chaining
     * @throws IllegalArgumentException if minAnnRows is not positive
     */
    public Builder minAnnRows(int minAnnRows) {
      if (minAnnRows <= 0) {
        throw new IllegalArgumentException("minAnnRows must be positive, got: " + minAnnRows);
      }
      this.minAnnRows = minAnnRows;
      return this;
    }

    /**
     * Sets whether to create an ANN index when extending the dataset.
     *
     * @param val true to create ANN index on extend, false otherwise
     * @return This Builder instance for method chaining
     */
    public Builder createAnnIndexOnExtend(boolean val) {
      this.createAnnIndexOnExtend = val;
      return this;
    }

    /**
     * Sets the CAGRA-specific parameters for the ANN algorithm.
     *
     * @param params The {@link CagraIndexParams} configuration for CAGRA
     *               algorithm
     * @return This Builder instance for method chaining
     * @throws NullPointerException if params is null
     */
    public Builder withCagraParams(CagraIndexParams params) {
      this.cagraParams = Objects.requireNonNull(params);
      return this;
    }

    /**
     * Builds and returns a {@link TieredIndexParams} instance with the
     * configured parameters.
     *
     * @return A new TieredIndexParams instance
     * @throws IllegalStateException if CAGRA params are required but not
     *                               provided
     */
    public TieredIndexParams build() {
      if (cagraParams == null) throw new IllegalStateException("CAGRA params required");
      return new TieredIndexParams(this);
    }
  }
}
