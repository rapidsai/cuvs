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
import com.nvidia.cuvs.spi.CuVSProvider;

/**
 * Scalar 8-bit quantizer implementation that transforms 32-bit float datasets into 8-bit integer datasets.
 *
 * <p>This quantizer reduces each float32 value to an 8-bit signed integer, providing significant
 * compression while maintaining reasonable precision. The quantizer requires training on a representative
 * dataset to compute optimal quantization parameters (min/max values per dimension).
 *
 * <p>The quantizer supports both transformation and inverse transformation, allowing approximate
 * recovery of the original float32 values from the quantized representation.
 *
 * @since 25.08
 */
public class Scalar8BitQuantizer implements CuVSQuantizer {
  private final Object impl;

  /**
   * The data type used by this quantizer (BYTE for 8-bit signed integers).
   */
  private final DataType outputDataType = DataType.BYTE;

  /**
   * Creates a new scalar 8-bit quantizer with CuVSMatrix training data.
   *
   * <p>The training dataset is used to compute quantization parameters (min/max values)
   * for each dimension. The quantizer will be immediately ready for use after construction.
   *
   * @param resources The CuVS resources to use for quantization operations
   * @param trainingDataset A CuVSMatrix containing training vectors (must have FLOAT data type)
   * @throws Throwable if an error occurs during quantizer training
   * @throws IllegalArgumentException if trainingDataset is null or doesn't have FLOAT data type
   */
  public Scalar8BitQuantizer(CuVSResources resources, CuVSMatrix trainingDataset) throws Throwable {
    if (trainingDataset == null) {
      throw new IllegalArgumentException("Training dataset cannot be null");
    }
    if (trainingDataset.dataType() != DataType.FLOAT) {
      throw new IllegalArgumentException(
          "Training dataset must have FLOAT data type, got " + trainingDataset.dataType());
    }
    this.impl = CuVSProvider.provider().createScalar8BitQuantizerImpl(resources, trainingDataset);
  }

  /**
   * Returns the data type of quantized data produced by this quantizer.
   *
   * @return The DataType (BYTE for this scalar quantizer)
   */
  @Override
  public DataType outputDataType() {
    return outputDataType;
  }

  @Override
  public CuVSMatrix transform(CuVSMatrix input) throws Throwable {
    // Validate input data type
    if (input.dataType() != DataType.FLOAT) {
      throw new IllegalArgumentException(
          "Scalar8BitQuantizer requires FLOAT input, got " + input.dataType());
    }

    CuVSMatrix result = CuVSProvider.provider().transformScalar8Bit(impl, input);

    // Validate output data type
    if (result.dataType() != DataType.BYTE) {
      throw new IllegalStateException(
          "Expected BYTE output from scalar quantization, got " + result.dataType());
    }

    return result;
  }

  @Override
  public void train(CuVSMatrix trainingData) throws Throwable {
    // Training is handled during construction for ScalarQuantizer
    // This method could be implemented if you want to support re-training
    throw new UnsupportedOperationException("Training handled during construction");
  }

  @Override
  public CuVSMatrix inverseTransform(CuVSMatrix quantizedData) throws Throwable {
    // Validate input data type for inverse transform
    if (quantizedData.dataType() != DataType.BYTE) {
      throw new IllegalArgumentException(
          "Inverse transform requires BYTE input, got " + quantizedData.dataType());
    }

    return CuVSProvider.provider().inverseTransformScalar8Bit(impl, quantizedData);
  }

  @Override
  public void close() throws Exception {
    try {
      CuVSProvider.provider().closeScalar8BitQuantizer(impl);
    } catch (Throwable t) {
      if (t instanceof Exception) {
        throw (Exception) t;
      } else {
        // Wrap non-Exception throwables (like Error) in RuntimeException
        throw new RuntimeException("Error closing Scalar8BitQuantizer", t);
      }
    }
  }
}
