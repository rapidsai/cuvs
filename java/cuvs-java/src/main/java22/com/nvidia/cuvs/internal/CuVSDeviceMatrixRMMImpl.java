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

import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class CuVSDeviceMatrixRMMImpl extends CuVSDeviceMatrixImpl implements CuVSDeviceMatrix {

  private final CuVSResources resources;

  public CuVSDeviceMatrixRMMImpl(
      CuVSResources resources, long size, long columns, DataType dataType, int copyType) {
    super(
        resources,
        allocateRMMSegment(resources, size, columns, valueLayoutFromType(dataType)),
        size,
        columns,
        dataType,
        valueLayoutFromType(dataType),
        copyType);
    this.resources = resources;
  }

  private static MemorySegment allocateRMMSegment(
      CuVSResources resources, long size, long columns, ValueLayout valueLayout) {
    try (var resourcesAccess = resources.access()) {
      return Util.allocateRMMSegment(
          resourcesAccess.handle(), size * columns * valueLayout.byteSize());
    }
  }

  @Override
  public void close() {
    super.close();
    var bytes = getMatrixSizeInBytes();
    try (var resourcesAccessor = resources.access()) {
      checkCuVSError(cuvsRMMFree(resourcesAccessor.handle(), memorySegment, bytes), "cuvsRMMFree");
    }
  }
}
