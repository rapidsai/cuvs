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
public interface CuVSMatrix extends AutoCloseable {

  enum DataType {
    FLOAT,
    INT,
    UINT,
    BYTE
  }

  enum MemoryKind {
    HOST,
    DEVICE
  }

  /**
   * Creates a dataset from an on-heap array of vectors.
   * This method will allocate an additional MemorySegment to hold the graph data.
   *
   * @since 25.08
   */
  static CuVSMatrix ofArray(float[][] vectors) {
    return CuVSProvider.provider().newMatrixFromArray(vectors);
  }

  /**
   * Creates a dataset from an on-heap array of vectors.
   * This method will allocate an additional MemorySegment to hold the graph data.
   *
   * @since 25.08
   */
  static CuVSMatrix ofArray(int[][] vectors) {
    return CuVSProvider.provider().newMatrixFromArray(vectors);
  }

  /**
   * Creates a dataset from an on-heap array of vectors.
   * This method will allocate an additional MemorySegment to hold the graph data.
   *
   * @since 25.08
   */
  static CuVSMatrix ofArray(byte[][] vectors) {
    return CuVSProvider.provider().newMatrixFromArray(vectors);
  }

  /**
   * A builder to construct a new matrix one row at a time
   * @param <T> the CuVSMatrix type to build
   */
  interface Builder<T extends CuVSMatrix> {
    /**
     * Adds a single vector to the matrix.
     *
     * @param vector A float array of as many elements as the dimensions
     */
    void addVector(float[] vector);

    /**
     * Adds a single vector to the matrix.
     *
     * @param vector A byte array of as many elements as the dimensions
     */
    void addVector(byte[] vector);

    /**
     * Adds a single vector to the matrix.
     *
     * @param vector An int array of as many elements as the dimensions
     */
    void addVector(int[] vector);

    T build();
  }

  /**
   * Returns a builder to create a new instance of a dataset
   *
   * @param size     Number of vectors in the dataset
   * @param columns  Size of each vector in the dataset
   * @param dataType The data type of the dataset elements
   * @return a builder for creating a {@link CuVSHostMatrix}
   */
  static Builder<CuVSHostMatrix> hostBuilder(long size, long columns, DataType dataType) {
    return CuVSProvider.provider().newHostMatrixBuilder(size, columns, dataType);
  }

  /**
   * Returns a builder to create a new instance of a dataset
   *
   * @param resources CuVS resources used to allocate the device memory needed
   * @param size      Number of vectors in the dataset
   * @param columns   Size of each vector in the dataset
   * @param dataType  The data type of the dataset elements
   * @return a builder for creating a {@link CuVSDeviceMatrix}
   */
  static Builder<CuVSDeviceMatrix> deviceBuilder(
      CuVSResources resources, long size, long columns, DataType dataType) {
    return CuVSProvider.provider().newDeviceMatrixBuilder(resources, size, columns, dataType);
  }

  /**
   * Gets the size of the dataset
   *
   * @return Size of the dataset
   */
  long size();

  /**
   * Gets the number of columns in the Dataset (e.g. the dimensions of the vectors in this dataset,
   * or the graph degree for the graph represented as a list of neighbours
   *
   * @return Dimensions of the vectors in the dataset
   */
  long columns();

  /**
   * Gets the element type
   *
   * @return a {@link DataType} describing the matrix element type
   */
  DataType dataType();

  /**
   * Get a view (0-copy) of the row data, as a list of integers (32 bit)
   *
   * @param row the row for which to return the data
   */
  RowView getRow(long row);

  /**
   * Copies the content of this dataset to an on-heap Java matrix (array of arrays).
   *
   * @param array the destination array. Must be of length {@link CuVSMatrix#size()} or bigger,
   *              and each element must be of length {@link CuVSMatrix#columns()} or bigger.
   */
  void toArray(int[][] array);

  /**
   * Copies the content of this dataset to an on-heap Java matrix (array of arrays).
   *
   * @param array the destination array. Must be of length {@link CuVSMatrix#size()} or bigger,
   *              and each element must be of length {@link CuVSMatrix#columns()} or bigger.
   */
  void toArray(float[][] array);

  /**
   * Copies the content of this dataset to an on-heap Java matrix (array of arrays).
   *
   * @param array the destination array. Must be of length {@link CuVSMatrix#size()} or bigger,
   *              and each element must be of length {@link CuVSMatrix#columns()} or bigger.
   */
  void toArray(byte[][] array);

  @Override
  void close();
}
