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

import com.nvidia.cuvs.spi.CuVSProvider;

/**
 * This represents a wrapper for a dataset to be used for index construction.
 * The purpose is to allow a caller to place the vectors into native memory
 * directly, instead of requiring the caller to load all the vectors into the heap
 * (e.g. with a float[][]).
 *
 * @since 25.06
 */
public interface Dataset extends AutoCloseable {

  /**
   * Creates a dataset from an on-heap array of vectors
   *
   * @since 25.08
   */
  static Dataset ofArray(float[][] vectors) {
    return CuVSProvider.provider().newArrayDataset(vectors);
  }

  interface Builder {
    /**
     * Add a single vector to the dataset.
     *
     * @param vector A float array of as many elements as the dimensions
     */
    void addVector(float[] vector);

    Dataset build();
  }

  /**
   * Returns a builder to create a new instance of a dataset
   *
   * @param size       Number of vectors in the dataset
   * @param dimensions Size of each vector in the dataset
   * @return new instance of {@link Dataset}
   */
  static Dataset.Builder builder(int size, int dimensions) {
    return CuVSProvider.provider().newDatasetBuilder(size, dimensions);
  }

  /**
   * Gets the size of the dataset
   *
   * @return Size of the dataset
   */
  int size();

  /**
   * Gets the dimensions of the vectors in this dataset
   *
   * @return Dimensions of the vectors in the dataset
   */
  int dimensions();
}
