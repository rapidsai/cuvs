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
 * Binary quantizer implementation that transforms 32-bit float datasets into 1-bit binary datasets.
 *
 * <p>Binary quantization reduces each float32 value to a single bit, providing maximum compression
 * but with significant precision loss. This quantizer does not require training and does not
 * support inverse transformation.
 *
 * @since 25.08
 */
public class BinaryQuantizer implements CuVSQuantizer {
  private final CuVSResources resources;
  private final int precision = 8;

  /**
   * Creates a new binary quantizer with the specified resources.
   *
   * @param resources The CuVS resources to use for quantization operations
   */
  public BinaryQuantizer(CuVSResources resources) {
    this.resources = resources;
    // Note: Binary quantization is stateless, no impl object needed
  }

  @Override
  public int precision() {
    return precision;
  }

  @Override
  public CuVSMatrix transform(CuVSMatrix input) throws Throwable {
    // Validate input precision
    if (input.precision() != 32) {
      throw new IllegalArgumentException(
          "BinaryQuantizer requires 32-bit float input, got " + input.precision() + "-bit");
    }

    // Use the correct provider method signature (resources, not impl)
    CuVSMatrix result = CuVSProvider.provider().transformBinary(resources, input);

    // Validate output precision
    if (result.precision() != 8) {
      throw new IllegalStateException(
          "Expected 1-bit output from binary quantization, got " + result.precision() + "-bit");
    }

    return result;
  }

  @Override
  public void train(CuVSMatrix trainingData) throws Throwable {
    // Binary quantization doesn't require training
    throw new UnsupportedOperationException("Binary quantization does not require training");
  }

  @Override
  public CuVSMatrix inverseTransform(CuVSMatrix quantizedData) throws Throwable {
    // Binary quantization typically doesn't support exact inverse transformation
    throw new UnsupportedOperationException(
        "Binary quantization does not support inverse transformation");
  }

  @Override
  public void close() throws Exception {
    // Binary quantizer is stateless, no cleanup needed
  }
}
