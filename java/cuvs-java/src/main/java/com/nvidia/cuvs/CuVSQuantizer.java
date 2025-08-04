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

import com.nvidia.cuvs.CuVSMatrix.DataType;

/**
 * Base interface for all cuVS quantizers providing unified quantization operations.
 *
 */
public interface CuVSQuantizer extends AutoCloseable {

  /**
   * Returns the data type of quantized data produced by this quantizer.
   *
   * @return The DataType of the quantized output (BYTE for quantized data)
   */
  DataType outputDataType();

  /**
   * Transforms a float32 dataset into a quantized dataset.
   *
   * @param input The input dataset of float32 vectors (must have precision() == 32)
   * @return A new Dataset containing the quantized vectors with this quantizer's precision
   */
  CuVSMatrix transform(CuVSMatrix input) throws Throwable;

  /**
   * Optional training method - only applies to quantizers that require training.
   *
   * @param trainingData The dataset to train on (must have precision() == 32)
   * @throws Throwable if an error occurs during training
   */
  default void train(CuVSMatrix trainingData) throws Throwable {
    throw new UnsupportedOperationException(
        "Training not supported for " + getClass().getSimpleName());
  }

  /**
   * Optional inverse transformation - only applies to quantizers that support reconstruction
   *
   * @param quantizedData The quantized dataset to inverse transform
   *                      (must have precision() matching this quantizer's precision())
   * @return A float[][] array containing the de-quantized vectors
   */
  default CuVSMatrix inverseTransform(CuVSMatrix quantizedData) throws Throwable {
    throw new UnsupportedOperationException(
        "Inverse transform not supported for " + getClass().getSimpleName());
  }

  /**
   * Closes this quantizer and releases any associated resources.
   *
   * @throws Exception if an error occurs while closing
   */
  @Override
  void close() throws Exception;
}
