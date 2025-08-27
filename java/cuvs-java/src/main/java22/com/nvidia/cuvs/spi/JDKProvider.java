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
package com.nvidia.cuvs.spi;

import static com.nvidia.cuvs.internal.common.Util.*;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;

final class JDKProvider implements CuVSProvider {

  private static final MethodHandle createNativeDataset$mh = createNativeDatasetBuilder();

  static MethodHandle createNativeDatasetBuilder() {
    try {
      var lookup = MethodHandles.lookup();
      var mt =
          MethodType.methodType(
              CuVSMatrix.class,
              MemorySegment.class,
              int.class,
              int.class,
              CuVSMatrix.DataType.class);
      return lookup.findStatic(JDKProvider.class, "createNativeDataset", mt);
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private static CuVSMatrix createNativeDataset(
      MemorySegment memorySegment, int size, int dimensions, CuVSMatrix.DataType dataType) {
    return new CuVSHostMatrixImpl(memorySegment, size, dimensions, dataType);
  }

  @Override
  public CuVSResources newCuVSResources(Path tempDirectory) {
    Objects.requireNonNull(tempDirectory);
    if (Files.notExists(tempDirectory)) {
      throw new IllegalArgumentException("does not exist:" + tempDirectory);
    }
    if (!Files.isDirectory(tempDirectory)) {
      throw new IllegalArgumentException("not a directory:" + tempDirectory);
    }
    return new CuVSResourcesImpl(tempDirectory);
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) {
    return BruteForceIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) {
    return CagraIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) {
    return HnswIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) {
    return TieredIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes) {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    return CagraIndexImpl.merge(indexes);
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes, CagraMergeParams mergeParams) {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    return CagraIndexImpl.merge(indexes, mergeParams);
  }

  @Override
  public GPUInfoProvider gpuInfoProvider() {
    return new GPUInfoProviderImpl();
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, CuVSMatrix.DataType dataType) throws UnsupportedOperationException {

    return new CuVSMatrix.Builder<>() {
      final CuVSHostMatrixArenaImpl matrix = new CuVSHostMatrixArenaImpl(size, columns, dataType);
      int current = 0;

      @Override
      public void addVector(float[] vector) {
        if (vector.length != columns) {
          throw new IllegalArgumentException(
              String.format(
                  Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
        }
        internalAddVector(vector);
      }

      @Override
      public void addVector(byte[] vector) {
        if (vector.length != columns) {
          throw new IllegalArgumentException(
              String.format(
                  Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
        }
        internalAddVector(vector);
      }

      @Override
      public void addVector(int[] vector) {
        if (vector.length != columns) {
          throw new IllegalArgumentException(
              String.format(
                  Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
        }
        internalAddVector(vector);
      }

      private void internalAddVector(Object vector) {
        if (current >= size) throw new ArrayIndexOutOfBoundsException();
        MemorySegment.copy(
            vector,
            0,
            matrix.memorySegment(),
            matrix.valueLayout(),
            ((current++) * columns * matrix.valueLayout().byteSize()),
            (int) columns);
      }

      @Override
      public CuVSHostMatrix build() {
        return matrix;
      }
    };
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType)
      throws UnsupportedOperationException {
    return new HeapSegmentBuilder(resources, size, columns, dataType);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    return new HeapSegmentBuilder(resources, size, columns, rowStride, columnStride, dataType);
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    return createNativeDataset$mh;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.FLOAT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.INT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.BYTE);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  /**
   * This {@link CuVSDeviceMatrix} builder implementation returns a {@link CuVSDeviceMatrix} backed by managed RMM
   * device memory. It uses a non-native {@link MemorySegment} created directly from on-heap java arrays to avoid
   * an intermediate allocation and copy to a native (off-heap) segment.
   * It requires the copy function ({@code cudaMemcpyAsync}) to have the {@code Critical} linker option in order
   * to allow the access to on-heap memory (see {@link Util#cudaMemcpyAsync}).
   */
  private static class HeapSegmentBuilder implements CuVSMatrix.Builder<CuVSDeviceMatrix> {
    private final long columns;
    private final long size;
    private final CuVSDeviceMatrixImpl matrix;
    private final MemorySegment stream;
    private int current;

    private HeapSegmentBuilder(
        CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType) {
      this.columns = columns;
      this.size = size;
      this.matrix = CuVSDeviceMatrixRMMImpl.create(resources, size, columns, dataType);
      this.stream = Util.getStream(resources);
      this.current = 0;
    }

    private HeapSegmentBuilder(
        CuVSResources resources,
        long size,
        long columns,
        int rowStride,
        int columnStride,
        CuVSMatrix.DataType dataType) {
      this.columns = columns;
      this.size = size;
      this.matrix =
          CuVSDeviceMatrixRMMImpl.create(
              resources, size, columns, rowStride, columnStride, dataType);
      this.stream = Util.getStream(resources);
      this.current = 0;
    }

    @Override
    public void addVector(float[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    @Override
    public void addVector(byte[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    @Override
    public void addVector(int[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    private void internalAddVector(MemorySegment vector) {
      if (current >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }

      long rowBytes = columns * matrix.valueLayout().byteSize();

      var dstOffset = ((current++) * rowBytes);
      var dst = matrix.memorySegment().asSlice(dstOffset);
      cudaMemcpyAsync(dst, vector, rowBytes, CudaMemcpyKind.HOST_TO_DEVICE, stream);
    }

    @Override
    public CuVSDeviceMatrix build() {
      return matrix;
    }
  }
}
