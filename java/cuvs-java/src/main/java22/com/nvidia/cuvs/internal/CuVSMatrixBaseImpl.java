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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_CHAR;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSMatrix;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public abstract class CuVSMatrixBaseImpl implements CuVSMatrix {
  protected final MemorySegment memorySegment;
  protected final DataType dataType;
  protected final long size;
  protected final long columns;

  protected CuVSMatrixBaseImpl(
      MemorySegment memorySegment, DataType dataType, long size, long columns) {
    this.memorySegment = memorySegment;
    this.dataType = dataType;
    this.size = size;
    this.columns = columns;
  }

  @Override
  public long size() {
    return size;
  }

  @Override
  public long columns() {
    return columns;
  }

  public MemorySegment memorySegment() {
    return memorySegment;
  }

  protected int bits() {
    return dataType.bytes() * 8;
  }

  protected int code() {
    return switch (dataType) {
      case FLOAT -> kDLFloat();
      case INT -> kDLInt();
      case UINT, BYTE -> kDLUInt();
    };
  }

  protected static ValueLayout valueLayoutFromType(DataType dataType) {
    return switch (dataType) {
      case FLOAT -> C_FLOAT;
      case INT, UINT -> C_INT;
      case BYTE -> C_CHAR;
    };
  }

  public abstract MemorySegment toTensor(Arena arena);
}
