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

import java.lang.foreign.Arena;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A Dataset implementation backed by host (CPU) memory.
 * Memory is allocated and managed by the implementation, via a Java shared {@link Arena}
 */
public class CuVSHostMatrixArenaImpl extends CuVSHostMatrixImpl {
  private final AtomicReference<Arena> arenaReference;

  public CuVSHostMatrixArenaImpl(long size, long columns, DataType dataType) {
    this(
        size,
        columns,
        -1,
        -1,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, -1, dataType),
        Arena.ofShared());
  }

  public CuVSHostMatrixArenaImpl(
      long size, long columns, int rowStride, int columnStride, DataType dataType) {
    this(
        size,
        columns,
        rowStride,
        columnStride,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, rowStride, dataType),
        Arena.ofShared());
  }

  private CuVSHostMatrixArenaImpl(
      long size,
      long columns,
      int rowStride,
      int columnStride,
      DataType dataType,
      ValueLayout valueLayout,
      SequenceLayout layout,
      Arena arena) {
    super(
        arena.allocate(layout),
        size,
        columns,
        rowStride,
        columnStride,
        dataType,
        valueLayout,
        layout);

    if (columnStride != -1) {
      throw new UnsupportedOperationException(
          "Stridden columns are currently not supported; columnStride must be equal to -1");
    }

    this.arenaReference = new AtomicReference<>(arena);
  }

  @Override
  public void close() {
    var arena = arenaReference.getAndSet(null);
    if (arena != null) {
      arena.close();
    }
  }
}
