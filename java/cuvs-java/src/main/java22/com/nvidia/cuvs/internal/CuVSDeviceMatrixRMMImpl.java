/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.internal.common.CloseableRMMAllocation;
import java.lang.foreign.ValueLayout;

/**
 * A Dataset implementation backed by device (GPU) memory, allocated
 * using RMM functions from {@link CuVSResources} pools (via {@link CloseableRMMAllocation}).
 */
public class CuVSDeviceMatrixRMMImpl extends CuVSDeviceMatrixImpl implements CuVSDeviceMatrix {

  private final CloseableRMMAllocation rmmAllocation;

  private CuVSDeviceMatrixRMMImpl(
      CuVSResources resources,
      CloseableRMMAllocation rmmAllocation,
      long size,
      long columns,
      DataType dataType,
      ValueLayout valueLayout) {
    super(resources, rmmAllocation.handle(), size, columns, dataType, valueLayout);
    this.rmmAllocation = rmmAllocation;
  }

  private CuVSDeviceMatrixRMMImpl(
      CuVSResources resources,
      CloseableRMMAllocation rmmAllocation,
      long size,
      long columns,
      long rowStride,
      long columnStride,
      DataType dataType,
      ValueLayout valueLayout) {
    super(
        resources,
        rmmAllocation.handle(),
        size,
        columns,
        rowStride,
        columnStride,
        dataType,
        valueLayout);
    this.rmmAllocation = rmmAllocation;
  }

  public static CuVSDeviceMatrixImpl create(
      CuVSResources resources, long size, long columns, DataType dataType) {
    try (var resourcesAccess = resources.access()) {
      var valueLayout = valueLayoutFromType(dataType);
      var rmmAllocation =
          CloseableRMMAllocation.allocateRMMSegment(
              resourcesAccess.handle(), size * columns * valueLayout.byteSize());
      return new CuVSDeviceMatrixRMMImpl(
          resources, rmmAllocation, size, columns, dataType, valueLayout);
    }
  }

  public static CuVSDeviceMatrixImpl create(
      CuVSResources resources,
      long size,
      long columns,
      long rowStride,
      long columnStride,
      DataType dataType) {

    var valueLayout = valueLayoutFromType(dataType);
    var elementSize = valueLayout.byteSize();

    final long rowSize;
    if (rowStride <= 0) {
      rowSize = columns * elementSize;
    } else if (rowStride >= columns) {
      rowSize = rowStride * elementSize;
    } else {
      throw new IllegalArgumentException("Row stride cannot be less than the number of columns");
    }
    if (columnStride != -1) {
      throw new UnsupportedOperationException(
          "Stridden columns are currently not supported; columnStride must be equal to -1");
    }

    try (var resourcesAccess = resources.access()) {

      var rmmAllocation =
          CloseableRMMAllocation.allocateRMMSegment(resourcesAccess.handle(), size * rowSize);
      return new CuVSDeviceMatrixRMMImpl(
          resources, rmmAllocation, size, columns, rowStride, columnStride, dataType, valueLayout);
    }
  }

  @Override
  public void close() {
    super.close();
    rmmAllocation.close();
  }
}
