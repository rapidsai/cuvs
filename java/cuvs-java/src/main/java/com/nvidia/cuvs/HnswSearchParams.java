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

import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.panama.CuVSHnswSearchParams;

/**
 * HnswSearchParams encapsulates the logic for configuring and holding search
 * parameters for HNSW index.
 *
 * @since 25.02
 */
public class HnswSearchParams {

  private CuVSResources resources;
  private MemorySegment memorySegment;
  private int ef = 200;
  private int numThreads = 0;

  /**
   * Constructs an instance of HnswSearchParams with passed search parameters.
   *
   * @param resources  the resources instance to use
   * @param ef         the ef value
   * @param numThreads the number of threads
   *
   */
  private HnswSearchParams(CuVSResources resources, int ef, int numThreads) {
    this.resources = resources;
    this.ef = ef;
    this.numThreads = numThreads;
    this.memorySegment = allocateMemorySegment();
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment allocateMemorySegment() {
    MemorySegment memorySegment = CuVSHnswSearchParams.allocate(resources.arena);
    CuVSHnswSearchParams.ef(memorySegment, ef);
    CuVSHnswSearchParams.num_threads(memorySegment, numThreads);
    return memorySegment;
  }

  public MemorySegment getHnswSearchParamsMemorySegment() {
    return memorySegment;
  }

  /**
   * Gets the ef value
   *
   * @return the integer ef value
   */
  public int getEf() {
    return ef;
  }

  /**
   * Gets the number of threads
   *
   * @return the number of threads
   */
  public int getNumThreads() {
    return numThreads;
  }

  @Override
  public String toString() {
    return "HnswSearchParams [ef=" + ef + ", numThreads=" + numThreads + "]";
  }

  /**
   * Builder configures and creates an instance of HnswSearchParams.
   */
  public static class Builder {

    private CuVSResources resources;
    private int ef = 200;
    private int numThreads = 0;

    /**
     * Constructs this Builder with an instance of Arena.
     *
     * @param resources the {@link CuVSResources} instance to use
     */
    public Builder(CuVSResources resources) {
      this.resources = resources;
    }

    /**
     * Sets the ef value
     *
     * @param ef the ef value
     * @return an instance of this Builder
     */
    public Builder withEF(int ef) {
      this.ef = ef;
      return this;
    }

    /**
     * Sets the number of threads
     *
     * @param numThreads the number of threads
     * @return an instance of this Builder
     */
    public Builder withNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    /**
     * Builds an instance of {@link HnswSearchParams} with passed search parameters.
     *
     * @return an instance of HnswSearchParams
     */
    public HnswSearchParams build() {
      return new HnswSearchParams(resources, ef, numThreads);
    }
  }
}
