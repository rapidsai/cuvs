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
 * HnswSearchParams encapsulates the logic for configuring and holding search
 * parameters for HNSW index.
 *
 * @param ef the ef value
 * @param numThreads the number of threads
 * @since 25.02
 */
public record HnswSearchParams (int ef, int numThreads) {

    public HnswSearchParams {
      if (ef < 0) {
        throw new IllegalArgumentException();
      }
      if (numThreads < 0) {
        throw new IllegalArgumentException();
      }
    }

  /**
   * Builder configures and creates an instance of HnswSearchParams.
   */
  public static class Builder {

    private static final int DEFAULT_EF_VALUE = 200;
    private static final int DEFAULT_NUM_THREADS = 0;

    private int ef = DEFAULT_EF_VALUE;
    private int numThreads = DEFAULT_NUM_THREADS;

    /**
     * Constructs this Builder with an instance of Arena.
     */
    public Builder() { }

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
      return new HnswSearchParams(ef, numThreads);
    }
  }
}
