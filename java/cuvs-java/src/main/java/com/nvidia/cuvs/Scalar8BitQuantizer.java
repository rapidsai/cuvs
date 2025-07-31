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
   * The bit precision used by this quantizer (8-bit signed integers).
   */
  private final int precision = 8;

  /**
   * Creates a new scalar 8-bit quantizer with CuVSMatrix training data.
   *
   * <p>The training dataset is used to compute quantization parameters (min/max values)
   * for each dimension. The quantizer will be immediately ready for use after construction.
   *
   * @param resources The CuVS resources to use for quantization operations
   * @param trainingDataset A CuVSMatrix containing training vectors (must have 32-bit precision)
   * @throws Throwable if an error occurs during quantizer training
   * @throws IllegalArgumentException if trainingDataset is null or doesn't have 32-bit precision
   */
  public Scalar8BitQuantizer(CuVSResources resources, CuVSMatrix trainingDataset) throws Throwable {
    if (trainingDataset == null) {
      throw new IllegalArgumentException("Training dataset cannot be null");
    }
    if (trainingDataset.precision() != 32) {
      throw new IllegalArgumentException(
          "Training dataset must have 32-bit precision, got "
              + trainingDataset.precision()
              + "-bit");
    }
    this.impl = CuVSProvider.provider().createScalar8BitQuantizerImpl(resources, trainingDataset);
  }

  /**
   * Returns the bit precision of quantized data produced by this quantizer.
   *
   * @return The bit precision (8 for this scalar quantizer)
   */
  @Override
  public int precision() {
    return precision;
  }

  @Override
  public CuVSMatrix transform(CuVSMatrix input) throws Throwable {
    // Validate input precision
    if (input.precision() != 32) {
      throw new IllegalArgumentException(
          "Scalar8BitQuantizer requires 32-bit float input, got " + input.precision() + "-bit");
    }

    CuVSMatrix result = CuVSProvider.provider().transformScalar8Bit(impl, input);

    // Validate output precision
    if (result.precision() != 8) {
      throw new IllegalStateException(
          "Expected 8-bit output from scalar quantization, got " + result.precision() + "-bit");
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
    // Validate input precision for inverse transform
    if (quantizedData.precision() != 8) {
      throw new IllegalArgumentException(
          "Inverse transform requires 8-bit input, got " + quantizedData.precision() + "-bit");
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
