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
  private final ThresholdType thresholdType;

  /**
   * The bit precision used by this quantizer (8-bit for packed binary data).
   */
  private final int precision = 8;

  /**
   * Creates a new binary quantizer with the specified resources.
   *
   * @param resources The CuVS resources to use for quantization operations
   */
  public BinaryQuantizer(CuVSResources resources) {
    this(resources, ThresholdType.ZERO);
  }

  /**
   * Creates a binary quantizer with specified threshold type.
   *
   * @param resources The CuVS resources
   * @param thresholdType The threshold type for binary conversion (ZERO, MEAN, or SAMPLING_MEDIAN)
   */
  public BinaryQuantizer(CuVSResources resources, ThresholdType thresholdType) {
    this.resources = resources;
    this.thresholdType = thresholdType;
  }

  /**
   * Threshold types supported by binary quantization.
   */
  public enum ThresholdType {
    /** Use zero as threshold value */
    ZERO(0),
    /** Use mean of dataset as threshold */
    MEAN(1),
    /** Use sampling median as threshold */
    SAMPLING_MEDIAN(2);

    private final int value;

    ThresholdType(int value) {
      this.value = value;
    }

    public int getValue() {
      return value;
    }
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

  /**
   * Gets the threshold value used for binary quantization.
   */
  public ThresholdType getThresholdType() {
    return thresholdType;
  }

  @Override
  public CuVSMatrix transform(CuVSMatrix input) throws Throwable {
    // Validate input precision
    if (input.precision() != 32) {
      throw new IllegalArgumentException(
          "BinaryQuantizer requires 32-bit float input, got " + input.precision() + "-bit");
    }

    CuVSMatrix result =
        CuVSProvider.provider().transformBinary(resources, input, thresholdType.getValue());

    // Validate output precision
    if (result.precision() != 8) {
      throw new IllegalStateException(
          "Expected 8-bit output from binary quantization, got " + result.precision() + "-bit");
    }

    return result;
  }

  @Override
  public void train(CuVSMatrix trainingData) throws Throwable {
    throw new UnsupportedOperationException(
        "Binary quantization performs training internally during each transform call");
  }

  @Override
  public CuVSMatrix inverseTransform(CuVSMatrix quantizedData) throws Throwable {
    throw new UnsupportedOperationException(
        "Binary quantization does not support inverse transformation");
  }

  @Override
  public void close() throws Exception {}
}
