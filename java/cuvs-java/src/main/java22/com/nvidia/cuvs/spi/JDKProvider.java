/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.panama.headers_h.CUVS_LOG_LEVEL_CRITICAL;
import static com.nvidia.cuvs.internal.panama.headers_h.CUVS_LOG_LEVEL_DEBUG;
import static com.nvidia.cuvs.internal.panama.headers_h.CUVS_LOG_LEVEL_ERROR;
import static com.nvidia.cuvs.internal.panama.headers_h.CUVS_LOG_LEVEL_INFO;
import static com.nvidia.cuvs.internal.panama.headers_h.CUVS_LOG_LEVEL_OFF;
import static com.nvidia.cuvs.internal.panama.headers_h.CUVS_LOG_LEVEL_TRACE;
import static com.nvidia.cuvs.internal.panama.headers_h.CUVS_LOG_LEVEL_WARN;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemcpy2DAsync;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsGetLogLevel;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsSetLogLevel;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsVersionGet;
import static com.nvidia.cuvs.internal.panama.headers_h.uint16_t;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaStreamSynchronize;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.PinnedMemoryBuffer;
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
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.logging.Level;

final class JDKProvider implements CuVSProvider {

  private static final MethodHandle createNativeDataset$mh;
  private static final MethodHandle createNativeDatasetWithStrides$mh;

  static {
    try {
      var lookup = MethodHandles.lookup();
      createNativeDataset$mh =
          lookup.findStatic(
              JDKProvider.class,
              "createNativeDataset",
              MethodType.methodType(
                  CuVSMatrix.class,
                  MemorySegment.class,
                  int.class,
                  int.class,
                  CuVSMatrix.DataType.class));

      createNativeDatasetWithStrides$mh =
          lookup.findStatic(
              JDKProvider.class,
              "createNativeDatasetWithStrides",
              MethodType.methodType(
                  CuVSMatrix.class,
                  MemorySegment.class,
                  int.class,
                  int.class,
                  int.class,
                  int.class,
                  CuVSMatrix.DataType.class));
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private final cuvsGetLogLevel GET_LOG_LEVEL_INVOKER = cuvsGetLogLevel.makeInvoker();

  private JDKProvider() {}

  static CuVSProvider create() throws ProviderInitializationException {
    NativeDependencyLoader.loadLibraries();

    var mavenVersion = readCuVSVersionFromManifest();

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
    }
    return new JDKProvider();
  }

  /**
   * Read cuvs-java version from this Jar Manifest, or null if these are not available
   */
  private static String readCuVSVersionFromManifest() {
    try (var jarFile =
        new JarFile(
            JDKProvider.class.getProtectionDomain().getCodeSource().getLocation().getPath())) {
      Manifest manifest = jarFile.getManifest();
      return manifest.getMainAttributes().getValue("Implementation-Version");
    } catch (IOException e) {
      return null;
    }
  }

  private static CuVSMatrix createNativeDataset(
      MemorySegment memorySegment, int size, int dimensions, CuVSMatrix.DataType dataType) {
    return new CuVSHostMatrixImpl(memorySegment, size, dimensions, dataType);
  }

  private static CuVSMatrix createNativeDatasetWithStrides(
      MemorySegment memorySegment,
      int size,
      int dimensions,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    return new CuVSHostMatrixImpl(
        memorySegment, size, dimensions, rowStride, columnStride, dataType);
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
  public void setLogLevel(Level level) {
    if (level.equals(Level.ALL) || level.equals(Level.FINEST)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_TRACE());
    } else if (level.equals(Level.FINER) || level.equals(Level.FINE)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_DEBUG());
    } else if (level.equals(Level.CONFIG) || level.equals(Level.INFO)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_INFO());
    } else if (level.equals(Level.WARNING)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_WARN());
    } else if (level.equals(Level.SEVERE)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_ERROR());
    } else if (level.equals(Level.OFF)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_OFF());
    } else {
      throw new UnsupportedOperationException("Unsupported log level [" + level + "]");
    }
  }

  @Override
  public Level getLogLevel() {
    int logLevel = GET_LOG_LEVEL_INVOKER.apply();
    if (logLevel == CUVS_LOG_LEVEL_TRACE()) {
      return Level.ALL;
    } else if (logLevel == CUVS_LOG_LEVEL_DEBUG()) {
      return Level.FINE;
    } else if (logLevel == CUVS_LOG_LEVEL_INFO()) {
      return Level.INFO;
    } else if (logLevel == CUVS_LOG_LEVEL_WARN()) {
      return Level.WARNING;
    } else if (logLevel == CUVS_LOG_LEVEL_ERROR() || logLevel == CUVS_LOG_LEVEL_CRITICAL()) {
      return Level.SEVERE;
    } else if (logLevel == CUVS_LOG_LEVEL_OFF()) {
      return Level.OFF;
    }
    throw new IllegalArgumentException("Unexpected log level [" + logLevel + "]");
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, CuVSMatrix.DataType dataType) {

    return new HostMatrixBuilder(size, columns, dataType);
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType) {

    return new HostMatrixBuilder(size, columns, rowStride, columnStride, dataType);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType) {
    return new DeviceMatrixBuilder(resources, size, columns, dataType);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    return new DeviceMatrixBuilder(resources, size, columns, rowStride, columnStride, dataType);
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    return createNativeDataset$mh;
  }

  @Override
  public MethodHandle newNativeMatrixBuilderWithStrides() {
    return createNativeDatasetWithStrides$mh;
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

  private abstract static class MatrixBuilder<T extends CuVSMatrixInternal> {

    protected final long columns;
    protected final long size;
    protected final T matrix;
    protected final long elementSize;
    protected final long rowSize;
    protected final long rowBytes;
    protected int currentRow;

    protected MatrixBuilder(T matrix, long size, long columns) {
      this.columns = columns;
      this.size = size;
      this.matrix = matrix;
      this.elementSize = matrix.valueLayout().byteSize();
      this.rowSize = columns * elementSize;
      this.rowBytes = rowSize;
      this.currentRow = 0;
    }

    protected MatrixBuilder(T matrix, long size, long columns, int rowStride) {
      this.columns = columns;
      this.size = size;
      this.matrix = matrix;
      this.elementSize = matrix.valueLayout().byteSize();
      this.rowSize = rowStride > 0 ? rowStride * elementSize : columns * elementSize;
      this.rowBytes = columns * elementSize;

      this.currentRow = 0;
    }

    public void addVector(float[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    public void addVector(byte[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    public void addVector(int[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    protected abstract void internalAddVector(MemorySegment vector);
  }

  /**
   * This {@link CuVSDeviceMatrix} builder implementation returns a {@link CuVSDeviceMatrix} backed by managed RMM
   * device memory. It uses a {@link PinnedMemoryBuffer} to batch data before copying it to the GPU.
   */
  private static final class DeviceMatrixBuilder extends MatrixBuilder<CuVSDeviceMatrixImpl>
      implements CuVSMatrix.Builder<CuVSDeviceMatrix> {

    private final MemorySegment stream;

    private final CuVSResources resources;
    private final long bufferRowCount;
    private int currentBufferRow;

    private DeviceMatrixBuilder(
        CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType) {
      super(CuVSDeviceMatrixRMMImpl.create(resources, size, columns, dataType), size, columns);
      this.stream = Util.getStream(resources);
      this.resources = resources;

      this.bufferRowCount = Math.min((PinnedMemoryBuffer.CHUNK_BYTES / rowBytes), size);
      this.currentBufferRow = 0;
    }

    private DeviceMatrixBuilder(
        CuVSResources resources,
        long size,
        long columns,
        int rowStride,
        int columnStride,
        CuVSMatrix.DataType dataType) {
      super(
          CuVSDeviceMatrixRMMImpl.create(
              resources, size, columns, rowStride, columnStride, dataType),
          size,
          columns,
          rowStride);

      this.stream = Util.getStream(resources);
      this.resources = resources;

      this.bufferRowCount = Math.min((PinnedMemoryBuffer.CHUNK_BYTES / rowBytes), size);
      this.currentBufferRow = 0;
    }

    @Override
    protected void internalAddVector(MemorySegment vector) {
      if (currentRow >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }
      var hostBufferOffset = currentBufferRow * rowBytes;
      try (var access = resources.access()) {
        var hostBuffer = CuVSResourcesImpl.getHostBuffer(access);
        MemorySegment.copy(vector, 0, hostBuffer, hostBufferOffset, rowBytes);

        currentRow++;
        currentBufferRow++;
        if (currentBufferRow == bufferRowCount) {
          flushBuffer(hostBuffer);
        }
      }
    }

    private void flushBuffer(MemorySegment hostBuffer) {
      if (currentBufferRow > 0) {
        var deviceMemoryOffset = (currentRow - currentBufferRow) * rowSize;
        var dst = matrix.memorySegment().asSlice(deviceMemoryOffset);
        checkCudaError(
            cudaMemcpy2DAsync(
                dst,
                rowSize,
                hostBuffer,
                rowBytes,
                rowBytes,
                currentBufferRow,
                CudaMemcpyKind.HOST_TO_DEVICE.kind,
                stream),
            "cudaMemcpy2DAsync");

        currentBufferRow = 0;
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      }
    }

    @Override
    public CuVSDeviceMatrix build() {
      try (var access = resources.access()) {
        var hostBuffer = CuVSResourcesImpl.getHostBuffer(access);
        flushBuffer(hostBuffer);
      }
      return matrix;
    }
  }

  private static class HostMatrixBuilder extends MatrixBuilder<CuVSHostMatrixImpl>
      implements CuVSMatrix.Builder<CuVSHostMatrix> {

    private HostMatrixBuilder(long size, long columns, CuVSMatrix.DataType dataType) {
      super(new CuVSHostMatrixArenaImpl(size, columns, dataType), size, columns);
    }

    private HostMatrixBuilder(
        long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType) {
      super(
          new CuVSHostMatrixArenaImpl(size, columns, rowStride, columnStride, dataType),
          size,
          columns,
          rowStride);
    }

    @Override
    protected void internalAddVector(MemorySegment vector) {
      if (currentRow >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }
      MemorySegment.copy(vector, 0, matrix.memorySegment(), ((currentRow++) * rowSize), rowBytes);
    }

    @Override
    public CuVSHostMatrix build() {
      return matrix;
    }
  }
}
