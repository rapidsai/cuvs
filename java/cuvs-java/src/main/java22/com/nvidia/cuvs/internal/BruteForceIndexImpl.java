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

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.HOST_TO_DEVICE;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.INFER_DIRECTION;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.concatenate;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBruteForceBuild;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBruteForceDeserialize;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBruteForceIndexCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBruteForceIndexDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBruteForceIndex_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBruteForceSearch;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBruteForceSerialize;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.omp_set_num_threads;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.common.CloseableRMMAllocation;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.BitSet;
import java.util.Objects;
import java.util.UUID;

/**
 *
 * {@link BruteForceIndex} encapsulates a BRUTEFORCE index, along with methods
 * to interact with it.
 *
 * @since 25.02
 */
public class BruteForceIndexImpl implements BruteForceIndex {

  private final CuVSResources resources;
  private final IndexReference bruteForceIndexReference;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   *
   * @param dataset               the dataset used for creating the BRUTEFORCE
   *                              index
   * @param resources             an instance of {@link CuVSResources}
   * @param bruteForceIndexParams an instance of {@link BruteForceIndexParams}
   *                              holding the index parameters
   */
  private BruteForceIndexImpl(
      CuVSMatrix dataset, CuVSResources resources, BruteForceIndexParams bruteForceIndexParams)
      throws Exception {
    Objects.requireNonNull(dataset);
    try (dataset) {
      this.resources = resources;
      assert dataset instanceof CuVSMatrixBaseImpl;
      this.bruteForceIndexReference = build((CuVSMatrixBaseImpl) dataset, bruteForceIndexParams);
    }
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private BruteForceIndexImpl(InputStream inputStream, CuVSResources resources) throws Throwable {
    this.resources = resources;
    this.bruteForceIndexReference = deserialize(inputStream);
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native destroy_brute_force_index function to de-allocate
   * BRUTEFORCE index
   */
  @Override
  public void close() {
    checkNotDestroyed();
    try {
      int returnValue = cuvsBruteForceIndexDestroy(bruteForceIndexReference.indexPtr);
      checkCuVSError(returnValue, "cuvsBruteForceIndexDestroy");
      bruteForceIndexReference.close(resources);
    } finally {
      destroyed = true;
    }
  }

  /**
   * Invokes the native build_brute_force_index function via the Panama API to
   * build the {@link BruteForceIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   * index
   */
  private IndexReference build(
      CuVSMatrixBaseImpl dataset, BruteForceIndexParams bruteForceIndexParams) {
    long rows = dataset.size();
    long cols = dataset.columns();

    MemorySegment datasetMemSegment = dataset.memorySegment();

    omp_set_num_threads(bruteForceIndexParams.getNumWriterThreads());

    long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
    var index = createBruteForceIndex();

    try (var resourcesAccessor = resources.access()) {
      long cuvsResources = resourcesAccessor.handle();
      try (var closeableDataMemorySegmentP = allocateRMMSegment(cuvsResources, datasetBytes)) {
        MemorySegment datasetMemorySegmentP = closeableDataMemorySegmentP.handle();

        cudaMemcpy(datasetMemorySegmentP, datasetMemSegment, datasetBytes, INFER_DIRECTION);

        long[] datasetShape = {rows, cols};
        var tensorDataArena = Arena.ofShared();
        MemorySegment datasetTensor =
            prepareTensor(tensorDataArena, datasetMemorySegmentP, datasetShape, 2, 32, 2, 1);

        var returnValue = cuvsStreamSync(cuvsResources);
        checkCuVSError(returnValue, "cuvsStreamSync");

        returnValue = cuvsBruteForceBuild(cuvsResources, datasetTensor, 0, 0.0f, index);
        checkCuVSError(returnValue, "cuvsBruteForceBuild");

        returnValue = cuvsStreamSync(cuvsResources);
        checkCuVSError(returnValue, "cuvsStreamSync");

        return new IndexReference(
            new CloseableRMMAllocation(closeableDataMemorySegmentP),
            datasetBytes,
            tensorDataArena,
            index);
      }
    } finally {
      omp_set_num_threads(1);
    }
  }

  /**
   * Invokes the native search_brute_force_index via the Panama API for searching
   * a BRUTEFORCE index.
   *
   * @param cuvsQuery an instance of {@link BruteForceQuery} holding the query
   *                  vectors and other parameters
   * @return an instance of {@link BruteForceSearchResults} containing the results
   */
  @Override
  public SearchResults search(BruteForceQuery cuvsQuery) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      checkNotDestroyed();
      long numQueries = cuvsQuery.getQueryVectors().length;
      long numBlocks = cuvsQuery.getTopK() * numQueries;
      int vectorDimension = numQueries > 0 ? cuvsQuery.getQueryVectors()[0].length : 0;

      SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
      SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsMemorySegment = localArena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = localArena.allocate(distancesSequenceLayout);

      // prepare the prefiltering data
      final long prefilterDataLength;
      final long prefilterBytes;
      final MemorySegment prefilterDataMemorySegment;
      BitSet[] prefilters = cuvsQuery.getPrefilters();
      if (prefilters != null && prefilters.length > 0) {
        BitSet concatenatedFilters = concatenate(prefilters, cuvsQuery.getNumDocs());
        long[] filters = concatenatedFilters.toLongArray();
        prefilterDataMemorySegment = buildMemorySegment(localArena, filters);
        prefilterDataLength = (long) cuvsQuery.getNumDocs() * prefilters.length;
        long[] prefilterShape = {(prefilterDataLength + 31) / 32};
        prefilterBytes = C_INT_BYTE_SIZE * prefilterShape[0];
      } else {
        prefilterDataLength = 0;
        prefilterBytes = 0;
        prefilterDataMemorySegment = MemorySegment.NULL;
      }

      MemorySegment querySeg = buildMemorySegment(localArena, cuvsQuery.getQueryVectors());

      int topk = cuvsQuery.getTopK();
      try (var resourcesAccessor = cuvsQuery.getResources().access()) {
        long cuvsResources = resourcesAccessor.handle();

        final long queriesBytes = C_FLOAT_BYTE_SIZE * numQueries * vectorDimension;
        final long neighborsBytes = C_LONG_BYTE_SIZE * numQueries * topk;
        final long distanceBytes = C_FLOAT_BYTE_SIZE * numQueries * topk;

        try (var queriesDP = allocateRMMSegment(cuvsResources, queriesBytes);
            var neighborsDP = allocateRMMSegment(cuvsResources, neighborsBytes);
            var distancesDP = allocateRMMSegment(cuvsResources, distanceBytes);
            var prefilterDP =
                prefilterBytes > 0
                    ? allocateRMMSegment(cuvsResources, prefilterBytes)
                    : CloseableRMMAllocation.EMPTY) {

          cudaMemcpy(queriesDP.handle(), querySeg, queriesBytes, INFER_DIRECTION);

          long[] queriesShape = {numQueries, vectorDimension};
          MemorySegment queriesTensor =
              prepareTensor(localArena, queriesDP.handle(), queriesShape, 2, 32, 2, 1);
          long[] neighborsShape = {numQueries, topk};
          MemorySegment neighborsTensor =
              prepareTensor(localArena, neighborsDP.handle(), neighborsShape, 0, 64, 2, 1);
          long[] distancesShape = {numQueries, topk};
          MemorySegment distancesTensor =
              prepareTensor(localArena, distancesDP.handle(), distancesShape, 2, 32, 2, 1);

          MemorySegment prefilter = cuvsFilter.allocate(localArena);
          MemorySegment prefilterTensor;

          if (prefilterDataMemorySegment == MemorySegment.NULL) {
            cuvsFilter.type(prefilter, 0); // NO_FILTER
            cuvsFilter.addr(prefilter, 0);
          } else {
            long[] prefilterShape = {(prefilterDataLength + 31) / 32};
            cudaMemcpy(
                prefilterDP.handle(), prefilterDataMemorySegment, prefilterBytes, HOST_TO_DEVICE);

            prefilterTensor =
                prepareTensor(localArena, prefilterDP.handle(), prefilterShape, 1, 32, 2, 1);

            cuvsFilter.type(prefilter, 2);
            cuvsFilter.addr(prefilter, prefilterTensor.address());
          }

          var returnValue = cuvsStreamSync(cuvsResources);
          checkCuVSError(returnValue, "cuvsStreamSync");

          returnValue =
              cuvsBruteForceSearch(
                  cuvsResources,
                  bruteForceIndexReference.indexPtr,
                  queriesTensor,
                  neighborsTensor,
                  distancesTensor,
                  prefilter);
          checkCuVSError(returnValue, "cuvsBruteForceSearch");

          returnValue = cuvsStreamSync(cuvsResources);
          checkCuVSError(returnValue, "cuvsStreamSync");

          cudaMemcpy(neighborsMemorySegment, neighborsDP.handle(), neighborsBytes, INFER_DIRECTION);
          cudaMemcpy(distancesMemorySegment, distancesDP.handle(), distanceBytes, INFER_DIRECTION);
        }
      }
      return BruteForceSearchResults.create(
          neighborsSequenceLayout,
          distancesSequenceLayout,
          neighborsMemorySegment,
          distancesMemorySegment,
          cuvsQuery.getTopK(),
          cuvsQuery.getMapping(),
          numQueries);
    }
  }

  @Override
  public void serialize(OutputStream outputStream) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".bf");
    serialize(outputStream, path);
  }

  @Override
  public void serialize(OutputStream outputStream, Path tempFile) throws Throwable {
    checkNotDestroyed();
    var tempFilePath = tempFile.toAbsolutePath();

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      int returnValue =
          cuvsBruteForceSerialize(
              resourcesAccessor.handle(),
              localArena.allocateFrom(tempFilePath.toString()),
              bruteForceIndexReference.indexPtr);
      checkCuVSError(returnValue, "cuvsBruteForceSerialize");
    }

    try (var inputStream = Files.newInputStream(tempFilePath)) {
      inputStream.transferTo(outputStream);
    } finally {
      Files.deleteIfExists(tempFile);
    }
  }

  private static MemorySegment createBruteForceIndex() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexPtrPtr = localArena.allocate(cuvsBruteForceIndex_t);
      // cuvsBruteForceIndexCreate gets a pointer to a cuvsBruteForceIndex_t, which is defined as a
      // pointer to cuvsBruteForceIndex.
      // It's basically an "out" parameter: the C functions will create the index and "return back"
      // a pointer to it.
      // The "out parameter" pointer is needed only for the duration of the function invocation (it
      // could be a stack pointer, in C) so we allocate it from our localArena, unwrap it and return
      // it.
      var returnValue = cuvsBruteForceIndexCreate(indexPtrPtr);
      checkCuVSError(returnValue, "cuvsBruteForceIndexCreate");
      return indexPtrPtr.get(cuvsBruteForceIndex_t, 0);
    }
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a BRUTEFORCE
   * index using an {@link InputStream}.
   *
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    checkNotDestroyed();
    Path tmpIndexFile =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".bf")
            .toAbsolutePath();
    IndexReference indexReference = new IndexReference(createBruteForceIndex());

    try (inputStream;
        var outputStream = Files.newOutputStream(tmpIndexFile);
        var arena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      inputStream.transferTo(outputStream);

      int returnValue =
          cuvsBruteForceDeserialize(
              resourcesAccessor.handle(),
              arena.allocateFrom(tmpIndexFile.toString()),
              indexReference.indexPtr);
      checkCuVSError(returnValue, "cuvsBruteForceDeserialize");

    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
    return indexReference;
  }

  public static BruteForceIndex.Builder newBuilder(CuVSResources cuvsResources) {
    return new Builder(Objects.requireNonNull(cuvsResources));
  }

  /**
   * Builder helps configure and create an instance of {@link BruteForceIndex}.
   */
  public static class Builder implements BruteForceIndex.Builder {

    private CuVSMatrix dataset;
    private final CuVSResources cuvsResources;
    private BruteForceIndexParams bruteForceIndexParams;
    private InputStream inputStream;

    /**
     * Constructs this Builder with an instance of {@link CuVSResources}.
     *
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    /**
     * Registers an instance of configured {@link BruteForceIndexParams} with this
     * Builder.
     *
     * @param bruteForceIndexParams An instance of BruteForceIndexParams
     * @return An instance of this Builder
     */
    @Override
    public Builder withIndexParams(BruteForceIndexParams bruteForceIndexParams) {
      this.bruteForceIndexParams = bruteForceIndexParams;
      return this;
    }

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     *
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     */
    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    /**
     * Sets the dataset for building the {@link BruteForceIndex}.
     *
     * @param vectors a two-dimensional float array
     * @return an instance of this Builder
     */
    @Override
    public Builder withDataset(float[][] vectors) {
      this.dataset = CuVSMatrix.ofArray(vectors);
      return this;
    }

    /**
     * Sets the dataset for building the {@link BruteForceIndex}.
     *
     * @param dataset a {@link CuVSMatrix} object containing the vectors
     * @return an instance of this Builder
     */
    @Override
    public Builder withDataset(CuVSMatrix dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Builds and returns an instance of {@link BruteForceIndex}.
     *
     * @return an instance of {@link BruteForceIndex}
     */
    @Override
    public BruteForceIndexImpl build() throws Throwable {
      if (inputStream != null) {
        return new BruteForceIndexImpl(inputStream, cuvsResources);
      } else {
        return new BruteForceIndexImpl(dataset, cuvsResources, bruteForceIndexParams);
      }
    }
  }

  /**
   * Holds the memory reference to a BRUTEFORCE index, its associated dataset, and the arena used to allocate
   * input data
   */
  private static class IndexReference {

    private final CloseableRMMAllocation datasetAllocationHandle;
    private final long datasetBytes;
    private final Arena tensorDataArena;
    private final MemorySegment indexPtr;

    private IndexReference(
        CloseableRMMAllocation datasetAllocationHandle,
        long datasetBytes,
        Arena tensorDataArena,
        MemorySegment indexPtr) {
      this.datasetAllocationHandle = datasetAllocationHandle;
      this.datasetBytes = datasetBytes;
      this.tensorDataArena = tensorDataArena;
      this.indexPtr = indexPtr;
    }

    private IndexReference(MemorySegment indexPtr) {
      this.datasetAllocationHandle = CloseableRMMAllocation.EMPTY;
      this.datasetBytes = 0;
      this.tensorDataArena = null;
      this.indexPtr = indexPtr;
    }

    /**
     * Free up the memory used for dataset, tensor-data.
     */
    private void close(CuVSResources resources) {
      try (var resourcesAccessor = resources.access()) {
        datasetAllocationHandle.close();
      }
      if (tensorDataArena != null) {
        tensorDataArena.close();
      }
    }
  }
}
