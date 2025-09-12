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

import static com.nvidia.cuvs.internal.common.NativeLibraryUtils.JVM_LoadLibrary$mh;
import static com.nvidia.cuvs.internal.common.NativeLibraryUtils.JVM_UnloadLibrary$mh;
import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsVersionGet;
import static com.nvidia.cuvs.internal.panama.headers_h.uint16_t;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.Util;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

final class JDKProvider implements CuVSProvider {

  static {
    OptionalNativeDependencyLoader.loadLibraries();
  }

  private static final MethodHandle createNativeDataset$mh = createNativeDatasetBuilder();

  static CuVSProvider create() throws Throwable {
    var mavenVersion = readMavenVersionFromPropertiesOrNull();

    try (var localArena = Arena.ofConfined()) {
      var majorPtr = localArena.allocate(uint16_t);
      var minorPtr = localArena.allocate(uint16_t);
      var patchPtr = localArena.allocate(uint16_t);
      checkCuVSError(cuvsVersionGet(majorPtr, minorPtr, patchPtr), "cuvsVersionGet");
      var major = majorPtr.get(uint16_t, 0);
      var minor = minorPtr.get(uint16_t, 0);
      var patch = patchPtr.get(uint16_t, 0);

      var cuvsVersionString = String.format(Locale.ROOT, "%02d.%02d.%d", major, minor, patch);
      if (mavenVersion != null && !cuvsVersionString.equals(mavenVersion)) {
        throw new ProviderInitializationException(
            String.format(
                Locale.ROOT,
                "libcuvs_c version mismatch: expected [%s], found [%s]",
                mavenVersion,
                cuvsVersionString));
      }
    } catch (ExceptionInInitializerError e) {
      if (e.getCause() instanceof IllegalArgumentException) {
        // Try to find if we failed to load libcuvs and why
        // jextract loads the dynamic library with SymbolLookup.libraryLookup; this uses
        // RawNativeLibraries::load
        // https://github.com/openjdk/jdk/blob/master/src/java.base/share/native/libjava/RawNativeLibraries.c#L58
        // RawNativeLibraries::load in turn calls JVM_LoadLibrary. Unfortunately, it calls it with a
        // JNI_FALSE parameter for throwException, which means that the detailed error messages are
        // not surfaced.
        // Here we try and load it again, with throwException true, so we can see what's broken
        try (var localArena = Arena.ofConfined()) {
          var name = localArena.allocateFrom(System.mapLibraryName("cuvs_c"));
          Object lib = JVM_LoadLibrary$mh.invoke(name, true);
          if (lib != null) {
            // It wasn't a problem with library loading, so undo what we did
            JVM_UnloadLibrary$mh.invoke(lib);
          }
        } catch (Throwable ex) {
          if (ex instanceof UnsatisfiedLinkError ulex) {
            throw new ProviderInitializationException(ulex.getMessage(), ulex);
          }
          throw new ProviderInitializationException("error while loading libcuvs", ex);
        }
      } else {
        throw e.getCause() != null ? e.getCause() : e;
      }
    }
    return new JDKProvider();
  }

  /**
   * Read cuvs-java version from Maven generated properties, or null if these are not available
   */
  private static String readMavenVersionFromPropertiesOrNull() {
    var properties = new Properties();

    try (var is = JDKProvider.class.getClassLoader().getResourceAsStream("version.properties")) {
      properties.load(is);
      return properties.getProperty("version");
    } catch (IOException e) {
      return null;
    }
  }

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
