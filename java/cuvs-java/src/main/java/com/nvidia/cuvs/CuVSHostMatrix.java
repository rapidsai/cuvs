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

/**
 * A Dataset implementation backed by host (CPU) memory.
 */
public interface CuVSHostMatrix extends CuVSMatrix {
  int get(int row, int col);

  default CuVSHostMatrix toHost() {
    return new CuVSHostMatrixDelegate(this);
  }

  default CuVSDeviceMatrix toDevice(CuVSResources resources) {
    var deviceMatrix = CuVSMatrix.deviceBuilder(resources, size(), columns(), dataType()).build();
    toDevice(deviceMatrix, resources);
    return deviceMatrix;
  }

  class CuVSHostMatrixDelegate implements CuVSHostMatrix {
    private final CuVSHostMatrix hostMatrix;

    public CuVSHostMatrixDelegate(CuVSHostMatrix cuVSHostMatrix) {
      this.hostMatrix = cuVSHostMatrix;
    }

    @Override
    public int get(int row, int col) {
      return hostMatrix.get(row, col);
    }

    @Override
    public long size() {
      return hostMatrix.size();
    }

    @Override
    public long columns() {
      return hostMatrix.columns();
    }

    @Override
    public DataType dataType() {
      return hostMatrix.dataType();
    }

    @Override
    public RowView getRow(long row) {
      return hostMatrix.getRow(row);
    }

    @Override
    public void toArray(int[][] array) {
      hostMatrix.toArray(array);
    }

    @Override
    public void toArray(float[][] array) {
      hostMatrix.toArray(array);
    }

    @Override
    public void toArray(byte[][] array) {
      hostMatrix.toArray(array);
    }

    @Override
    public MemoryKind memoryKind() {
      return hostMatrix.memoryKind();
    }

    @Override
    public void toHost(CuVSHostMatrix hostMatrix) {
      this.hostMatrix.toHost(hostMatrix);
    }

    @Override
    public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources) {
      hostMatrix.toDevice(deviceMatrix, cuVSResources);
    }

    @Override
    public void close() {
      // Do nothing
    }
  }
}
