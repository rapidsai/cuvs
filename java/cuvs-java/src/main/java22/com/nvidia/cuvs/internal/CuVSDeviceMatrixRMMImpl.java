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
    try (var resourcesAccess = resources.access()) {
      var valueLayout = valueLayoutFromType(dataType);
      var rmmAllocation =
          CloseableRMMAllocation.allocateRMMSegment(
              resourcesAccess.handle(), size * columns * valueLayout.byteSize());
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
