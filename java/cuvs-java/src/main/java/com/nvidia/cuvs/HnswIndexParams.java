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

/**
 * Supplemental parameters to build HNSW index.
 *
 * @since 25.02
 */
public class HnswIndexParams {

  /**
   * Hierarchy for HNSW index when converting from CAGRA index
   *
   * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only
   * index.
   */
  public enum CuvsHnswHierarchy {

    /**
     * Flat hierarchy, search is base-layer only
     */
    NONE(0),

    /**
     * Full hierarchy is built using the CPU
     */
    CPU(1);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CuvsHnswHierarchy(int value) {
      this.value = value;
    }
  };

  private CuvsHnswHierarchy hierarchy = CuvsHnswHierarchy.NONE;
  private int efConstruction = 200;
  private int numThreads = 2;
  private int vectorDimension;

  private HnswIndexParams(
      CuvsHnswHierarchy hierarchy, int efConstruction, int numThreads, int vectorDimension) {
    this.hierarchy = hierarchy;
    this.efConstruction = efConstruction;
    this.numThreads = numThreads;
    this.vectorDimension = vectorDimension;
  }

  /**
   *
   * @return
   */
  public CuvsHnswHierarchy getHierarchy() {
    return hierarchy;
  }

  /**
   *
   * @return
   */
  public int getEfConstruction() {
    return efConstruction;
  }

  /**
   *
   * @return
   */
  public int getNumThreads() {
    return numThreads;
  }

  /**
   *
   * @return
   */
  public int getVectorDimension() {
    return vectorDimension;
  }

  @Override
  public String toString() {
    return "HnswIndexParams [hierarchy="
        + hierarchy
        + ", efConstruction="
        + efConstruction
        + ", numThreads="
        + numThreads
        + ", vectorDimension="
        + vectorDimension
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link HnswIndexParams}.
   */
  public static class Builder {

    private CuvsHnswHierarchy hierarchy = CuvsHnswHierarchy.NONE;
    private int efConstruction = 200;
    private int numThreads = 2;
    private int vectorDimension;

    /**
     * Constructs this Builder with an instance of Arena.
     */
    public Builder() {}

    /**
     * Sets the hierarchy for HNSW index when converting from CAGRA index.
     *
     * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only
     * index.
     *
     * @param hierarchy the hierarchy for HNSW index when converting from CAGRA
     *                  index
     * @return an instance of Builder
     */
    public Builder withHierarchy(CuvsHnswHierarchy hierarchy) {
      this.hierarchy = hierarchy;
      return this;
    }

    /**
     * Sets the size of the candidate list during hierarchy construction when
     * hierarchy is `CPU`.
     *
     * @param efConstruction the size of the candidate list during hierarchy
     *                       construction when hierarchy is `CPU`
     * @return an instance of Builder
     */
    public Builder withEfConstruction(int efConstruction) {
      this.efConstruction = efConstruction;
      return this;
    }

    /**
     * Sets the number of host threads to use to construct hierarchy when hierarchy
     * is `CPU`.
     *
     * @param numThreads the number of threads
     * @return an instance of Builder
     */
    public Builder withNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    /**
     * Sets the vector dimension
     *
     * @param vectorDimension the vector dimension
     * @return an instance of Builder
     */
    public Builder withVectorDimension(int vectorDimension) {
      this.vectorDimension = vectorDimension;
      return this;
    }

    /**
     * Builds an instance of {@link HnswIndexParams}.
     *
     * @return an instance of {@link HnswIndexParams}
     */
    public HnswIndexParams build() {
      return new HnswIndexParams(hierarchy, efConstruction, numThreads, vectorDimension);
    }
  }
}
