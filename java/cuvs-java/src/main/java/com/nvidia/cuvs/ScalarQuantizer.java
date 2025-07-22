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
import java.util.Objects;

/**
 * {@link ScalarQuantizer} provides scalar quantization functionality for datasets.
 *
 * Scalar quantization maps floating-point values to 8-bit integers using learned
 * quantization parameters (scale and offset) that minimize quantization error.
 */
public interface ScalarQuantizer extends AutoCloseable {

  /**
   * Applies scalar quantization transform to given dataset.
   *
   * @param dataset a two-dimensional float array to transform
   * @return a QuantizedMatrix containing the scalar quantized data
   */
  QuantizedMatrix transform(float[][] dataset) throws Throwable;

  /**
   * Applies scalar quantization transform to given dataset.
   *
   * @param dataset a {@link Dataset} object containing the vectors to transform
   * @return a QuantizedMatrix containing the scalar quantized data
   */
  QuantizedMatrix transform(Dataset dataset) throws Throwable;

  /**
   * Applies inverse scalar quantization transform to given quantized dataset.
   *
   * @param quantizedData a two-dimensional byte array to inverse transform
   * @return a two-dimensional float array containing the dequantized data
   */
  float[][] inverseTransform(byte[][] quantizedData) throws Throwable;

  /**
   * Applies inverse scalar quantization transform to given quantized dataset.
   *
   * @param quantizedMatrix a QuantizedMatrix containing quantized data
   * @return a two-dimensional float array containing the dequantized data
   */
  float[][] inverseTransform(QuantizedMatrix quantizedMatrix) throws Throwable;

  /**
   * Convenience method that applies scalar quantization and immediately converts to array.
   *
   * @param dataset a two-dimensional float array to transform
   * @return a two-dimensional byte array containing the scalar quantized data
   */
  default byte[][] transformToArray(float[][] dataset) throws Throwable {
    try (QuantizedMatrix matrix = transform(dataset)) {
      return matrix.toArray();
    }
  }

  /**
   * Convenience method that applies scalar quantization and immediately converts to array.
   *
   * @param dataset a {@link Dataset} object containing the vectors to transform
   * @return a two-dimensional byte array containing the scalar quantized data
   */
  default byte[][] transformToArray(Dataset dataset) throws Throwable {
    try (QuantizedMatrix matrix = transform(dataset)) {
      return matrix.toArray();
    }
  }

  /**
   * Destroys the scalar quantizer and frees associated resources.
   */
  void destroy() throws Throwable;

  @Override
  default void close() throws Exception {
    destroy();
  }

  /**
   * Creates a new ScalarQuantizer builder.
   *
   * @param cuvsResources the CuVS resources to use
   * @return a new ScalarQuantizer.Builder instance
   */
  static Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    return CuVSProvider.provider().newScalarQuantizerBuilder(cuvsResources);
  }

  /**
   * Builder for creating ScalarQuantizer instances.
   */
  interface Builder {

    /**
     * Sets the quantile value for quantization.
     *
     * @param quantile the quantile value (must be between 0.0 and 1.0)
     * @return this builder
     */
    Builder withQuantile(float quantile);

    /**
     * Sets the training dataset for learning quantization parameters.
     *
     * @param trainingDataset a two-dimensional float array for training
     * @return this builder
     */
    Builder withTrainingDataset(float[][] trainingDataset);

    /**
     * Sets the training dataset for learning quantization parameters.
     *
     * @param trainingDataset a Dataset object for training
     * @return this builder
     */
    Builder withTrainingDataset(Dataset trainingDataset);

    /**
     * Builds the ScalarQuantizer instance.
     *
     * @return a new ScalarQuantizer instance
     * @throws Throwable if an error occurs during building
     */
    ScalarQuantizer build() throws Throwable;
  }
}
