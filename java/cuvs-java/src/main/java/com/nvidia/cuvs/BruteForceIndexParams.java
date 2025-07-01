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
 * Supplemental parameters to build BRUTEFORCE index.
 *
 * @since 25.02
 */
public class BruteForceIndexParams {

  private final int numWriterThreads;

  private BruteForceIndexParams(int writerThreads) {
    this.numWriterThreads = writerThreads;
  }

  @Override
  public String toString() {
    return "BruteForceIndexParams [numWriterThreads=" + numWriterThreads + "]";
  }

  /**
   * Gets the number of threads used to build the index.
   */
  public int getNumWriterThreads() {
    return numWriterThreads;
  }

  /**
   * Builder configures and creates an instance of {@link BruteForceIndexParams}.
   */
  public static class Builder {

    private int numWriterThreads = 2;

    /**
     * Sets the number of writer threads to use for indexing.
     *
     * @param numWriterThreads number of writer threads to use
     * @return an instance of Builder
     */
    public Builder withNumWriterThreads(int numWriterThreads) {
      this.numWriterThreads = numWriterThreads;
      return this;
    }

    /**
     * Builds an instance of {@link BruteForceIndexParams}.
     *
     * @return an instance of {@link BruteForceIndexParams}
     */
    public BruteForceIndexParams build() {
      return new BruteForceIndexParams(numWriterThreads);
    }
  }
}
