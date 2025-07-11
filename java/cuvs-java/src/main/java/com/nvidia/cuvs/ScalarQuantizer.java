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
 * {@link ScalarQuantizer} encapsulates a scalar quantizer for quantizing datasets.
 *
 * Scalar quantization is performed by a linear mapping of an interval in the
 * float data type to the full range of the quantized int type.
 */
public interface ScalarQuantizer {

  /**
   * De-allocate the scalar quantizer.
   */
  void destroy() throws Throwable;

  /**
   * Applies quantization transform to given dataset.
   *
   * @param dataset a two-dimensional float array to transform
   * @return a two-dimensional byte array containing the quantized data
   */
  byte[][] transform(float[][] dataset) throws Throwable;

  /**
   * Applies quantization transform to given dataset.
   *
   * @param dataset a {@link Dataset} object containing the vectors to transform
   * @return a two-dimensional byte array containing the quantized data
   */
  byte[][] transform(Dataset dataset) throws Throwable;

  /**
   * Perform inverse quantization step on previously quantized dataset.
   *
   * @param quantizedData a two-dimensional byte array of quantized data
   * @return a two-dimensional float array containing the dequantized data
   */
  float[][] inverseTransform(byte[][] quantizedData) throws Throwable;

  /**
   * Creates a new Builder with an instance of {@link CuVSResources}.
   *
   * @param cuvsResources an instance of {@link CuVSResources}
   * @throws UnsupportedOperationException if the provider does not support scalar quantization
   */
  static Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    return CuVSProvider.provider().newScalarQuantizerBuilder(cuvsResources);
  }

  /**
   * Builder helps configure and create an instance of {@link ScalarQuantizer}.
   */
  interface Builder {

    /**
     * Sets the quantile parameter for the scalar quantizer.
     * Specifies how many outliers at top and bottom will be ignored.
     * Must be within range of (0, 1].
     *
     * @param quantile the quantile value (default: 0.99)
     * @return an instance of this Builder
     */
    Builder withQuantile(float quantile);

    /**
     * Sets the training dataset for the scalar quantizer.
     *
     * @param dataset a two-dimensional float array for training
     * @return an instance of this Builder
     */
    Builder withTrainingDataset(float[][] dataset);

    /**
     * Sets the training dataset for the scalar quantizer.
     *
     * @param dataset a {@link Dataset} object containing the training vectors
     * @return an instance of this Builder
     */
    Builder withTrainingDataset(Dataset dataset);

    /**
     * Builds and returns an instance of {@link ScalarQuantizer}.
     *
     * @return an instance of {@link ScalarQuantizer}
     */
    ScalarQuantizer build() throws Throwable;
  }
}
