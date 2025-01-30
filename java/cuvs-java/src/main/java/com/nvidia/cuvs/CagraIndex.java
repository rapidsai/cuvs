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

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

import com.nvidia.cuvs.spi.CuVSProvider;

/**
 * {@link CagraIndex} encapsulates a CAGRA index, along with methods to interact
 * with it.
 * <p>
 * CAGRA is a graph-based nearest neighbors algorithm that was built from the
 * ground up for GPU acceleration. CAGRA demonstrates state-of-the art index
 * build and query performance for both small and large-batch sized search. Know
 * more about this algorithm
 * <a href="https://arxiv.org/abs/2308.15136" target="_blank">here</a>
 *
 * @since 25.02
 */
public interface CagraIndex {

    /**
     * Invokes the native destroy_cagra_index to de-allocate the CAGRA index
     */
    void destroyIndex() throws Throwable;

    /**
     * Invokes the native search_cagra_index via the Panama API for searching a
     * CAGRA index.
     *
     * @param query an instance of {@link CagraQuery} holding the query vectors and
     *              other parameters
     * @return an instance of {@link SearchResults} containing the results
     */
    SearchResults search(CagraQuery query) throws Throwable;

    /**
     * A method to persist a CAGRA index using an instance of {@link OutputStream}
     * for writing index bytes.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes into
     */
    void serialize(OutputStream outputStream) throws Throwable;

    /**
     * A method to persist a CAGRA index using an instance of {@link OutputStream}
     * for writing index bytes.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes into
     * @param bufferLength the length of buffer to use for writing bytes. Default
     *                     value is 1024
     */
    void serialize(OutputStream outputStream, int bufferLength) throws Throwable;

    /**
     * A method to persist a CAGRA index using an instance of {@link OutputStream}
     * for writing index bytes.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes into
     * @param tempFile     an intermediate {@link Path} where CAGRA index is written
     *                     temporarily
     */
    default void serialize(OutputStream outputStream, Path tempFile) throws Throwable {
        serialize(outputStream, tempFile, 1024);
    }

    /**
     * A method to persist a CAGRA index using an instance of {@link OutputStream}
     * and path to the intermediate temporary file.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes to
     * @param tempFile     an intermediate {@link Path} where CAGRA index is written
     *                     temporarily
     * @param bufferLength the length of buffer to use for writing bytes. Default
     *                     value is 1024
     */
    void serialize(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable;

    /**
     * A method to create and persist HNSW index from CAGRA index using an instance
     * of {@link OutputStream} and path to the intermediate temporary file.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes to
     */
    void serializeToHNSW(OutputStream outputStream) throws Throwable;

    /**
     * A method to create and persist HNSW index from CAGRA index using an instance
     * of {@link OutputStream} and path to the intermediate temporary file.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes to
     * @param bufferLength the length of buffer to use for writing bytes. Default
     *                     value is 1024
     */
    void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable;

    /**
     * A method to create and persist HNSW index from CAGRA index using an instance
     * of {@link OutputStream} and path to the intermediate temporary file.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes to
     * @param tempFile     an intermediate {@link Path} where CAGRA index is written
     *                     temporarily
     */
    default void serializeToHNSW(OutputStream outputStream, Path tempFile) throws Throwable {
        serializeToHNSW(outputStream, tempFile, 1024);
    }

    /**
     * A method to create and persist HNSW index from CAGRA index using an instance
     * of {@link OutputStream} and path to the intermediate temporary file.
     *
     * @param outputStream an instance of {@link OutputStream} to write the index
     *                     bytes to
     * @param tempFile     an intermediate {@link Path} where CAGRA index is written
     *                     temporarily
     * @param bufferLength the length of buffer to use for writing bytes. Default
     *                     value is 1024
     */
    void serializeToHNSW(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable;

    /**
     * Gets an instance of {@link CagraIndexParams}
     *
     * @return an instance of {@link CagraIndexParams}
     */
    CagraIndexParams getCagraIndexParameters();

    /**
     * Gets an instance of {@link CuVSResources}
     *
     * @return an instance of {@link CuVSResources}
     */
    CuVSResources getCuVSResources();

    /**
     * Creates a new Builder with an instance of {@link CuVSResources}.
     *
     * @param cuvsResources an instance of {@link CuVSResources}
     * @throws UnsupportedOperationException if the provider does not cuvs
     */
    static Builder newBuilder(CuVSResources cuvsResources) {
        Objects.requireNonNull(cuvsResources);
        return CuVSProvider.provider().newCagraIndexBuilder(cuvsResources);
    }

    /**
     * Builder helps configure and create an instance of {@link CagraIndex}.
     */
    interface Builder {

        /**
         * Sets an instance of InputStream typically used when index deserialization is
         * needed.
         *
         * @param inputStream an instance of {@link InputStream}
         * @return an instance of this Builder
         */
        Builder from(InputStream inputStream);

        /**
         * Sets the dataset for building the {@link CagraIndex}.
         *
         * @param dataset a two-dimensional float array
         * @return an instance of this Builder
         */
        Builder withDataset(float[][] dataset);

        /**
         * Registers an instance of configured {@link CagraIndexParams} with this
         * Builder.
         *
         * @param cagraIndexParameters An instance of CagraIndexParams.
         * @return An instance of this Builder.
         */
        Builder withIndexParams(CagraIndexParams cagraIndexParameters);

        /**
         * Registers an instance of configured {@link CagraCompressionParams} with this
         * Builder.
         *
         * @param cagraCompressionParams An instance of CagraCompressionParams.
         * @return An instance of this Builder.
         */
        public Builder withCompressionParams(CagraCompressionParams cagraCompressionParams);

        /**
         * Builds and returns an instance of CagraIndex.
         *
         * @return an instance of CagraIndex
         */
         CagraIndex build() throws Throwable;
    }
}
