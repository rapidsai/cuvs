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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaFreeHost;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaMallocHost;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.Arena;
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
  public CuVSMatrix.Builder newHostMatrixBuilder(
      int size, int columns, CuVSMatrix.DataType dataType) throws UnsupportedOperationException {

    return new CuVSMatrix.Builder() {
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
            columns);
      }

      @Override
      public CuVSMatrix build() {
        return matrix;
      }
    };
  }

  @Override
  public CuVSMatrix.Builder newDeviceMatrixBuilder(
      CuVSResources resources, int size, int columns, CuVSMatrix.DataType dataType, int copyType)
      throws UnsupportedOperationException {

    var builderCopyType = copyType & 0xF0;
    return switch (copyType) {
      case 0x10 -> new HeapSegmentBuilder(resources, size, columns, dataType, copyType);
      case 0x20 -> new CudaHostSegmentBuilder(resources, size, columns, dataType, copyType);
      default -> new NativeSegmentBuilder(resources, size, columns, dataType, copyType);
    };
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

  private static class NativeSegmentBuilder implements CuVSMatrix.Builder {
    private final int columns;
    private final int size;
    private final CuVSDeviceMatrixRMMImpl matrix;
    int current;
    MemorySegment tempSegment;
    final Arena tempSegmentArena;

    public NativeSegmentBuilder(
        CuVSResources resources,
        int size,
        int columns,
        CuVSMatrix.DataType dataType,
        int copyType) {
      this.columns = columns;
      this.size = size;
      this.matrix = new CuVSDeviceMatrixRMMImpl(resources, size, columns, dataType, copyType);
      current = 0;
      tempSegmentArena = Arena.ofShared();
    }

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
      if (current >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }

      long rowBytes = columns * matrix.valueLayout().byteSize();
      if (tempSegment == null) {
        tempSegment = tempSegmentArena.allocate(rowBytes);
      }

      MemorySegment.copy(vector, 0, tempSegment, matrix.valueLayout(), 0, columns);

      var dstOffset = ((current++) * rowBytes);
      var dst = matrix.memorySegment().asSlice(dstOffset);
      cudaMemcpy(dst, tempSegment, rowBytes);
    }

    @Override
    public CuVSMatrix build() {
      tempSegmentArena.close();
      return matrix;
    }
  }

  private static class HeapSegmentBuilder implements CuVSMatrix.Builder {
    private final int columns;
    private final int size;
    private final CuVSDeviceMatrixRMMImpl matrix;
    int current;

    public HeapSegmentBuilder(
        CuVSResources resources,
        int size,
        int columns,
        CuVSMatrix.DataType dataType,
        int copyType) {
      this.columns = columns;
      this.size = size;
      this.matrix = new CuVSDeviceMatrixRMMImpl(resources, size, columns, dataType, copyType);
      current = 0;
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
      cudaMemcpy(dst, vector, rowBytes);
    }

    @Override
    public CuVSMatrix build() {
      return matrix;
    }
  }

  private static class CudaHostSegmentBuilder implements CuVSMatrix.Builder {
    private final int columns;
    private final int size;
    private final CuVSDeviceMatrixRMMImpl matrix;
    int current;
    MemorySegment tempSegment;

    public CudaHostSegmentBuilder(
        CuVSResources resources,
        int size,
        int columns,
        CuVSMatrix.DataType dataType,
        int copyType) {
      this.columns = columns;
      this.size = size;
      this.matrix = new CuVSDeviceMatrixRMMImpl(resources, size, columns, dataType, copyType);
      current = 0;
    }

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

    private MemorySegment createBuffer(long bufferBytes) {
      try (var localArena = Arena.ofConfined()) {
        MemorySegment pointer = localArena.allocate(C_POINTER);
        cudaMallocHost(pointer, bufferBytes);
        return pointer.get(C_POINTER, 0);
      }
    }

    private void internalAddVector(Object vector) {
      if (current >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }

      long rowBytes = columns * matrix.valueLayout().byteSize();
      if (tempSegment == null) {
        tempSegment = createBuffer(rowBytes);
      }

      MemorySegment.copy(vector, 0, tempSegment, matrix.valueLayout(), 0, columns);

      var dstOffset = ((current++) * rowBytes);
      var dst = matrix.memorySegment().asSlice(dstOffset);
      cudaMemcpy(dst, tempSegment, rowBytes);
    }

    @Override
    public CuVSMatrix build() {
      if (tempSegment != null) {
        cudaFreeHost(tempSegment);
      }
      return matrix;
    }
  }
}
