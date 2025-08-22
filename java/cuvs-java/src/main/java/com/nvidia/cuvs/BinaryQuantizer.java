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

/** Binary quantizer that transforms float datasets into packed binary datasets. */
public class BinaryQuantizer implements CuVSQuantizer {
  private final CuVSResources resources;
  private final ThresholdType thresholdType;

  /** Output data type (BYTE for packed binary data). */
  private final DataType outputDataType = DataType.BYTE;

  private final Object impl;

  /** Creates a binary quantizer with training data. */
  public BinaryQuantizer(CuVSResources resources, CuVSMatrix trainingDataset) throws Throwable {
    this(resources, trainingDataset, ThresholdType.MEAN);
  }

  /** Creates a binary quantizer with specified threshold type and training data. */
  public BinaryQuantizer(
      CuVSResources resources, CuVSMatrix trainingDataset, ThresholdType thresholdType)
      throws Throwable {
    if (trainingDataset == null) {
      throw new IllegalArgumentException("Training dataset cannot be null");
    }
    if (trainingDataset.dataType() != DataType.FLOAT) {
      throw new IllegalArgumentException(
          "Training dataset must have FLOAT data type, got " + trainingDataset.dataType());
    }
    this.resources = resources;
    this.thresholdType = thresholdType;
    this.impl =
        CuVSProvider.provider()
            .createBinaryQuantizerImpl(resources, trainingDataset, thresholdType);
  }

  /** Threshold types for binary quantization. */
  public enum ThresholdType {
    /** Zero threshold */
    ZERO(0),
    /** Mean threshold */
    MEAN(1),
    /** Sampling median threshold */
    SAMPLING_MEDIAN(2);

    private final int value;

    ThresholdType(int value) {
      this.value = value;
    }

    public int getValue() {
      return value;
    }
  }

  /** Returns the output data type (BYTE). */
  @Override
  public DataType outputDataType() {
    return outputDataType;
  }

  /** Gets the threshold type used for binary quantization. */
  public ThresholdType getThresholdType() {
    return thresholdType;
  }

  @Override
  public CuVSMatrix transform(CuVSMatrix input) throws Throwable {
    if (input.dataType() != DataType.FLOAT) {
      throw new IllegalArgumentException(
          "BinaryQuantizer requires FLOAT input, got " + input.dataType());
    }

    CuVSMatrix result = CuVSProvider.provider().transformBinaryWithImpl(impl, input);

    if (result.dataType() != DataType.BYTE) {
      throw new IllegalStateException(
          "Expected BYTE output from binary quantization, got " + result.dataType());
    }

    return result;
  }

  @Override
  public void train(CuVSMatrix trainingData) throws Throwable {
    throw new UnsupportedOperationException("Training handled during construction");
  }

  @Override
  public CuVSMatrix inverseTransform(CuVSMatrix quantizedData) throws Throwable {
    throw new UnsupportedOperationException(
        "Binary quantization does not support inverse transformation");
  }

  @Override
  public void close() throws Exception {
    try {
      CuVSProvider.provider().closeBinaryQuantizer(impl);
    } catch (Throwable t) {
      if (t instanceof Exception) {
        throw (Exception) t;
      } else {
        throw new RuntimeException("Error closing BinaryQuantizer", t);
      }
    }
  }
}
