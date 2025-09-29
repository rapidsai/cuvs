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
import java.io.InputStream;
import java.util.Objects;

/**
 * {@link TieredIndex} encapsulates a Tiered index, along with methods to
 * interact with it.
 */
public interface TieredIndex extends AutoCloseable {

  /**
   * Destroys the underlying native TieredIndex object and releases associated
   * resources.
   *
   * @throws Exception if an error occurs during index destruction
   */
  @Override
  void close() throws Exception;

  /**
   * Searches the index with the specified query and search parameters.
   *
   * @param query An instance of {@link TieredIndexQuery} describing the queries
   *              and search parameters
   * @return An instance of {@link SearchResults} containing the k-nearest
   *         neighbors and their distances for each query
   * @throws Throwable if an error occurs during the search operation
   */
  SearchResults search(TieredIndexQuery query) throws Throwable;

  /**
   * Returns the algorithm type backing this TieredIndex.
   *
   * @return The {@link TieredIndexType} indicating the underlying algorithm
   *         (e.g., CAGRA)
   */
  TieredIndexType getIndexType();

  /**
   * Returns the resources handle associated with this TieredIndex.
   *
   * @return The {@link CuVSResources} instance used by this index
   */
  CuVSResources getCuVSResources();

  /**
   * Creates a new Builder with an instance of {@link CuVSResources}.
   *
   * @param cuvsResources An instance of {@link CuVSResources}
   * @return A new {@link Builder} instance for constructing a TieredIndex
   * @throws NullPointerException if cuvsResources is null
   */
  static Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    return CuVSProvider.provider().newTieredIndexBuilder(cuvsResources);
  }

  /**
   * Returns an ExtendBuilder to add new data to the existing index.
   *
   * @return An {@link ExtendBuilder} instance for extending the index
   */
  ExtendBuilder extend();

  /**
   * Builder interface for constructing {@link TieredIndex} instances.
   */
  interface Builder {

    /**
     *
     * @param inputStream The input stream containing serialized index data
     * @return This Builder instance for method chaining
     * @throws UnsupportedOperationException as deserialization is not yet
     *                                       supported
     */
    Builder from(InputStream inputStream);

    /**
     * Sets the dataset vectors for building the TieredIndex.
     *
     * @param vectors A two-dimensional float array containing the dataset
     *                vectors [n_vectors, dimensions]
     * @return This Builder instance for method chaining
     */
    Builder withDataset(float[][] vectors);

    /**
     * Sets the dataset for building the TieredIndex.
     *
     * @param dataset A {@link CuVSMatrix} instance containing the vectors
     * @return This Builder instance for method chaining
     */
    Builder withDataset(CuVSMatrix dataset);

    /**
     * Registers TieredIndex parameters with this Builder.
     *
     * @param params An instance of {@link TieredIndexParams} containing the
     *               index configuration
     * @return This Builder instance for method chaining
     */
    Builder withIndexParams(TieredIndexParams params);

    /**
     * Sets the index type for the TieredIndex.
     *
     * @param indexType The {@link TieredIndexType} to use (currently only CAGRA
     *                  is supported)
     * @return This Builder instance for method chaining
     */
    Builder withIndexType(TieredIndexType indexType);

    /**
     * Builds and returns an instance of TieredIndex with the configured
     * parameters.
     *
     * @return A new {@link TieredIndex} instance
     * @throws Throwable                if an error occurs during index
     *                                  construction
     * @throws IllegalArgumentException if both vectors and dataset are provided,
     *                                  or if required parameters are missing
     */
    TieredIndex build() throws Throwable;
  }

  /**
   * Enumeration of supported TieredIndex algorithm types.
   */
  enum TieredIndexType {
    CAGRA
  }

  /**
   * Builder interface for extending existing {@link TieredIndex} instances with
   * new data.
   */
  interface ExtendBuilder {

    /**
     * Sets the vectors to add to the existing index.
     *
     * @param vectors A two-dimensional float array containing the new vectors to
     *                add [n_new_vectors, dimensions]
     * @return This ExtendBuilder instance for method chaining
     */
    ExtendBuilder withDataset(float[][] vectors);

    /**
     * Sets the dataset to add to the existing index.
     *
     * @param dataset A {@link CuVSMatrix} instance containing the new vectors to
     *                add
     * @return This ExtendBuilder instance for method chaining
     */
    ExtendBuilder withDataset(CuVSMatrix dataset);

    /**
     * Executes the extend operation, adding the specified data to the index.
     *
     * @throws Throwable                if an error occurs during the extend
     *                                  operation
     * @throws IllegalArgumentException if both vectors and dataset are provided,
     *                                  or if no data is provided
     */
    void execute() throws Throwable;
  }
}
