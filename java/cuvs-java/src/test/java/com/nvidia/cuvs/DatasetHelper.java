/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

public class DatasetHelper {

  private static final MethodHandle createDataset$mh =
      CuVSProvider.provider().newNativeMatrixBuilder();
  private static final MethodHandle createDatasetWithStrides$mh =
      CuVSProvider.provider().newNativeMatrixBuilderWithStrides();

  public static CuVSMatrix fromMemorySegment(
      MemorySegment memorySegment, int size, int dimensions, CuVSMatrix.DataType dataType) {
    try {
      return (CuVSMatrix) createDataset$mh.invokeExact(memorySegment, size, dimensions, dataType);
    } catch (Throwable e) {
      if (e instanceof Error err) {
        throw err;
      } else if (e instanceof RuntimeException re) {
        throw re;
      } else {
        throw new RuntimeException(e);
      }
    }
  }

  public static CuVSMatrix fromMemorySegment(
      MemorySegment memorySegment,
      int size,
      int dimensions,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    try {
      return (CuVSMatrix)
          createDatasetWithStrides$mh.invokeExact(
              memorySegment, size, dimensions, rowStride, columnStride, dataType);
    } catch (Throwable e) {
      if (e instanceof Error err) {
        throw err;
      } else if (e instanceof RuntimeException re) {
        throw re;
      } else {
        throw new RuntimeException(e);
      }
    }
  }
}
