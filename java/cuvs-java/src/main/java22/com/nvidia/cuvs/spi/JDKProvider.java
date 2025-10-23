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

import static com.nvidia.cuvs.internal.CuVSParamsHelper.*;
import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.panama.headers_h.*;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaStreamSynchronize;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.common.PinnedMemoryBuffer;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqSearchParams;
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
  public CagraIndexParams cagraIndexParamsFromHnswHardM(
      long rows, long dim, int m, int efConstruction, CagraIndexParams.CuvsDistanceType metric) {
    try (var nativeCagraIndexParams = createCagraIndexParams();
        var ivfPqIndexParams = createIvfPqIndexParams();
        var ivfPqSearchParams = createIvfPqSearchParams()) {

      // This is already allocated by cuvsCagraIndexParamsCreate,
      // we just need to populate it.
      MemorySegment cuvsIvfPqParamsMemorySegment =
          cuvsCagraIndexParams.graph_build_params(nativeCagraIndexParams.handle());
      cuvsIvfPqParams.ivf_pq_build_params(cuvsIvfPqParamsMemorySegment, ivfPqIndexParams.handle());
      cuvsIvfPqParams.ivf_pq_search_params(
          cuvsIvfPqParamsMemorySegment, ivfPqSearchParams.handle());

      cuvsCagraIndexParams.graph_build_params(
          nativeCagraIndexParams.handle(), cuvsIvfPqParamsMemorySegment);
      checkCuVSError(
          cuvsCagraIndexParamsFromHnswHardM(
              nativeCagraIndexParams.handle(), rows, dim, m, efConstruction, metric.value),
          "cuvsCagraIndexParamsFromHnswHardM");

      return populateCagraIndexParamsFromNative(
          nativeCagraIndexParams,
          ivfPqIndexParams,
          ivfPqSearchParams,
          cuvsIvfPqParamsMemorySegment);
    }
  }

  @Override
  public CagraIndexParams cagraIndexParamsFromHnswSoftM(
      long rows, long dim, int m, int efConstruction, CagraIndexParams.CuvsDistanceType metric) {
    try (var nativeCagraIndexParams = createCagraIndexParams();
        var ivfPqIndexParams = createIvfPqIndexParams();
        var ivfPqSearchParams = createIvfPqSearchParams()) {

      // This is already allocated by cuvsCagraIndexParamsCreate,
      // we just need to populate it.
      MemorySegment cuvsIvfPqParamsMemorySegment =
          cuvsCagraIndexParams.graph_build_params(nativeCagraIndexParams.handle());
      cuvsIvfPqParams.ivf_pq_build_params(cuvsIvfPqParamsMemorySegment, ivfPqIndexParams.handle());
      cuvsIvfPqParams.ivf_pq_search_params(
          cuvsIvfPqParamsMemorySegment, ivfPqSearchParams.handle());

      cuvsCagraIndexParams.graph_build_params(
          nativeCagraIndexParams.handle(), cuvsIvfPqParamsMemorySegment);
      checkCuVSError(
          cuvsCagraIndexParamsFromHnswSoftM(
              nativeCagraIndexParams.handle(), rows, dim, m, efConstruction, metric.value),
          "cuvsCagraIndexParamsFromHnswSoftM");

      return populateCagraIndexParamsFromNative(
          nativeCagraIndexParams,
          ivfPqIndexParams,
          ivfPqSearchParams,
          cuvsIvfPqParamsMemorySegment);
    }
  }

  private static CagraIndexParams populateCagraIndexParamsFromNative(
      CloseableHandle nativeCagraIndexParams,
      CloseableHandle ivfPqIndexParams,
      CloseableHandle ivfPqSearchParams,
      MemorySegment cuvsIvfPqParamsMemorySegment) {
    var algo =
        CagraIndexParams.CagraGraphBuildAlgo.of(
            cuvsCagraIndexParams.build_algo(nativeCagraIndexParams.handle()));
    var builder =
        new CagraIndexParams.Builder()
            .withMetric(
                CagraIndexParams.CuvsDistanceType.of(
                    cuvsCagraIndexParams.metric(nativeCagraIndexParams.handle())))
            .withIntermediateGraphDegree(
                cuvsCagraIndexParams.intermediate_graph_degree(nativeCagraIndexParams.handle()))
            .withGraphDegree(cuvsCagraIndexParams.graph_degree(nativeCagraIndexParams.handle()))
            .withCagraGraphBuildAlgo(algo);

    if (algo == CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT) {
      builder.withNNDescentNumIterations(
          cuvsCagraIndexParams.nn_descent_niter(nativeCagraIndexParams.handle()));
    } else if (algo == CagraIndexParams.CagraGraphBuildAlgo.IVF_PQ) {
      builder.withCuVSIvfPqParams(
          new CuVSIvfPqParams.Builder()
              .withCuVSIvfPqIndexParams(
                  new CuVSIvfPqIndexParams.Builder()
                      .withMaxTrainPointsPerPqCode(
                          cuvsIvfPqIndexParams.max_train_points_per_pq_code(
                              ivfPqIndexParams.handle()))
                      .withAddDataOnBuild(
                          cuvsIvfPqIndexParams.add_data_on_build(ivfPqIndexParams.handle()))
                      .withMetric(
                          CagraIndexParams.CuvsDistanceType.of(
                              cuvsIvfPqIndexParams.metric(ivfPqIndexParams.handle())))
                      .withMetricArg(cuvsIvfPqIndexParams.metric_arg(ivfPqIndexParams.handle()))
                      .withNLists(cuvsIvfPqIndexParams.n_lists(ivfPqIndexParams.handle()))
                      .withKmeansNIters(
                          cuvsIvfPqIndexParams.kmeans_n_iters(ivfPqIndexParams.handle()))
                      .withKmeansTrainsetFraction(
                          cuvsIvfPqIndexParams.kmeans_trainset_fraction(ivfPqIndexParams.handle()))
                      .withPqBits(cuvsIvfPqIndexParams.pq_bits(ivfPqIndexParams.handle()))
                      .withPqDim(cuvsIvfPqIndexParams.pq_dim(ivfPqIndexParams.handle()))
                      .withCodebookKind(
                          CagraIndexParams.CodebookGen.of(
                              cuvsIvfPqIndexParams.codebook_kind(ivfPqIndexParams.handle())))
                      .withForceRandomRotation(
                          cuvsIvfPqIndexParams.force_random_rotation(ivfPqIndexParams.handle()))
                      .withConservativeMemoryAllocation(
                          cuvsIvfPqIndexParams.conservative_memory_allocation(
                              ivfPqIndexParams.handle()))
                      .build())
              .withCuVSIvfPqSearchParams(
                  new CuVSIvfPqSearchParams.Builder()
                      .withNProbes(cuvsIvfPqSearchParams.n_probes(ivfPqSearchParams.handle()))
                      .withLutDtype(
                          CagraIndexParams.CudaDataType.of(
                              cuvsIvfPqSearchParams.lut_dtype(ivfPqSearchParams.handle())))
                      .withInternalDistanceDtype(
                          CagraIndexParams.CudaDataType.of(
                              cuvsIvfPqSearchParams.internal_distance_dtype(
                                  ivfPqSearchParams.handle())))
                      .withPreferredShmemCarveout(
                          cuvsIvfPqSearchParams.preferred_shmem_carveout(
                              ivfPqSearchParams.handle()))
                      .build())
              .withRefinementRate(cuvsIvfPqParams.refinement_rate(cuvsIvfPqParamsMemorySegment))
              .build());
    }
    return builder.build();
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

    private final PinnedMemoryBuffer hostBuffer;
    private final long bufferRowCount;
    private int currentBufferRow;

    private DeviceMatrixBuilder(
        CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType) {
      super(CuVSDeviceMatrixRMMImpl.create(resources, size, columns, dataType), size, columns);
      this.stream = Util.getStream(resources);

      this.hostBuffer = new PinnedMemoryBuffer(size, columns, matrix.valueLayout());
      this.bufferRowCount = Math.min((hostBuffer.size() / rowBytes), size);
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

      this.hostBuffer = new PinnedMemoryBuffer(size, columns, matrix.valueLayout());
      this.bufferRowCount = Math.min((hostBuffer.size() / rowBytes), size);
      this.currentBufferRow = 0;
    }

    @Override
    protected void internalAddVector(MemorySegment vector) {
      if (currentRow >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }
      var hostBufferOffset = currentBufferRow * rowBytes;
      MemorySegment.copy(vector, 0, hostBuffer.address(), hostBufferOffset, rowBytes);

      currentRow++;
      currentBufferRow++;
      if (currentBufferRow == bufferRowCount) {
        flushBuffer();
      }
    }

    private void flushBuffer() {
      if (currentBufferRow > 0) {
        var deviceMemoryOffset = (currentRow - currentBufferRow) * rowSize;
        var dst = matrix.memorySegment().asSlice(deviceMemoryOffset);
        checkCudaError(
            cudaMemcpy2DAsync(
                dst,
                rowSize,
                hostBuffer.address(),
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
      flushBuffer();
      hostBuffer.close();
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
