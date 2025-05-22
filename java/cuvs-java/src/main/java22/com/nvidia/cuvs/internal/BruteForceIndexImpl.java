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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.common.Util.concatenate;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsBruteForceBuild;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsBruteForceDeserialize;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsBruteForceIndexCreate;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsBruteForceIndexDestroy;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsBruteForceIndex_t;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsBruteForceSearch;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsBruteForceSerialize;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsRMMAlloc;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsRMMFree;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsResources_t;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsStreamGet;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.omp_set_num_threads;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI_1.cudaMemcpy;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI_1.cudaStream_t;

import java.io.FileInputStream;
import java.io.FileOutputStream;
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

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.panama.cuvsBruteForceIndex;
import com.nvidia.cuvs.internal.panama.cuvsFilter;

/**
 *
 * {@link BruteForceIndex} encapsulates a BRUTEFORCE index, along with methods
 * to interact with it.
 *
 * @since 25.02
 */
public class BruteForceIndexImpl implements BruteForceIndex {

  private final float[][] dataset;
  private final CuVSResourcesImpl resources;
  private final IndexReference bruteForceIndexReference;
  private final BruteForceIndexParams bruteForceIndexParams;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   *
   * @param dataset               the dataset used for creating the BRUTEFORCE
   *                              index
   * @param resources             an instance of {@link CuVSResourcesImpl}
   * @param bruteForceIndexParams an instance of {@link BruteForceIndexParams}
   *                              holding the index parameters
   */
  private BruteForceIndexImpl(float[][] dataset, CuVSResourcesImpl resources,
      BruteForceIndexParams bruteForceIndexParams) throws Throwable {
    this.dataset = dataset;
    this.resources = resources;
    this.bruteForceIndexParams = bruteForceIndexParams;
    this.bruteForceIndexReference = build();
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResourcesImpl}
   */
  private BruteForceIndexImpl(InputStream inputStream, CuVSResourcesImpl resources) throws Throwable {
    this.bruteForceIndexParams = null;
    this.dataset = null;
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
  public void destroyIndex() throws Throwable {
    checkNotDestroyed();
    try {
      int returnValue = cuvsBruteForceIndexDestroy(bruteForceIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsBruteForceIndexDestroy");
    } finally {
      destroyed = true;
    }
  }

  /**
   * Invokes the native build_brute_force_index function via the Panama API to
   * build the {@link BruteForceIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      long rows = dataset.length;
      long cols = rows > 0 ? dataset[0].length : 0;

      Arena arena = resources.getArena();
      MemorySegment datasetMemSegment = buildMemorySegment(arena, dataset);

      long cuvsResources = resources.getMemorySegment().get(cuvsResources_t, 0);
      MemorySegment stream = arena.allocate(cudaStream_t);
      var returnValue = cuvsStreamGet(cuvsResources, stream);
      checkCuVSError(returnValue, "cuvsStreamGet");

      omp_set_num_threads(bruteForceIndexParams.getNumWriterThreads());

      MemorySegment datasetMemorySegment = arena.allocate(C_POINTER);

      long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
      returnValue = cuvsRMMAlloc(cuvsResources, datasetMemorySegment, datasetBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      // IMPORTANT: this should only come AFTER cuvsRMMAlloc call
      MemorySegment datasetMemorySegmentP = datasetMemorySegment.get(C_POINTER, 0);

      returnValue = cudaMemcpy(datasetMemorySegmentP, datasetMemSegment, datasetBytes, 4);
      checkCudaError(returnValue, "cudaMemcpy");

      long datasetShape[] = { rows, cols };
      MemorySegment datasetTensor = prepareTensor(arena, datasetMemorySegmentP, datasetShape, 2, 32, 2, 2, 1);

      MemorySegment index = arena.allocate(cuvsBruteForceIndex_t);

      returnValue = cuvsBruteForceIndexCreate(index);
      checkCuVSError(returnValue, "cuvsBruteForceIndexCreate");

      returnValue = cuvsStreamSync(cuvsResources);
      checkCuVSError(returnValue, "cuvsStreamSync");

      returnValue = cuvsBruteForceBuild(cuvsResources, datasetTensor, 0, 0.0f, index);
      checkCuVSError(returnValue, "cuvsBruteForceBuild");

      returnValue = cuvsStreamSync(cuvsResources);
      checkCuVSError(returnValue, "cuvsStreamSync");

      omp_set_num_threads(1);

      return new IndexReference(index);
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
      Arena arena = resources.getArena();

      SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
      SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsMemorySegment = arena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = arena.allocate(distancesSequenceLayout);

      // prepare the prefiltering data
      long prefilterDataLength = 0;
      MemorySegment prefilterDataMemorySegment = MemorySegment.NULL;
      BitSet[] prefilters = cuvsQuery.getPrefilters();
      if (prefilters != null && prefilters.length > 0) {
        BitSet concatenatedFilters = concatenate(prefilters, cuvsQuery.getNumDocs());
        long filters[] = concatenatedFilters.toLongArray();
        prefilterDataMemorySegment = buildMemorySegment(arena, filters);
        prefilterDataLength = cuvsQuery.getNumDocs() * prefilters.length;
      }

      MemorySegment querySeg = buildMemorySegment(arena, cuvsQuery.getQueryVectors());

      int topk = cuvsQuery.getTopK();
      long cuvsResources = resources.getMemorySegment().get(cuvsResources_t, 0);
      MemorySegment stream = arena.allocate(cudaStream_t);
      var returnValue = cuvsStreamGet(cuvsResources, stream);
      checkCuVSError(returnValue, "cuvsStreamGet");

      MemorySegment queriesD = arena.allocate(C_POINTER);
      MemorySegment neighborsD = arena.allocate(C_POINTER);
      MemorySegment distancesD = arena.allocate(C_POINTER);
      MemorySegment prefilterD = arena.allocate(C_POINTER);
      MemorySegment prefilterDP = MemorySegment.NULL;
      long prefilterLen = 0;

      long queriesBytes = C_FLOAT_BYTE_SIZE * numQueries * vectorDimension;
      long neighborsBytes = C_LONG_BYTE_SIZE * numQueries * topk;
      long distanceBytes = C_FLOAT_BYTE_SIZE * numQueries * topk;
      long prefilterBytes = 0; // size assigned later

      returnValue = cuvsRMMAlloc(cuvsResources, queriesD, queriesBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");
      returnValue = cuvsRMMAlloc(cuvsResources, neighborsD, neighborsBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");
      returnValue = cuvsRMMAlloc(cuvsResources, distancesD, distanceBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      // IMPORTANT: these three should only come AFTER cuvsRMMAlloc calls
      MemorySegment queriesDP = queriesD.get(C_POINTER, 0);
      MemorySegment neighborsDP = neighborsD.get(C_POINTER, 0);
      MemorySegment distancesDP = distancesD.get(C_POINTER, 0);

      returnValue = cudaMemcpy(queriesDP, querySeg, queriesBytes, 4);
      checkCudaError(returnValue, "cudaMemcpy");

      long queriesShape[] = { numQueries, vectorDimension };
      MemorySegment queriesTensor = prepareTensor(arena, queriesDP, queriesShape, 2, 32, 2, 2, 1);
      long neighborsShape[] = { numQueries, topk };
      MemorySegment neighborsTensor = prepareTensor(arena, neighborsDP, neighborsShape, 0, 64, 2, 2, 1);
      long distancesShape[] = { numQueries, topk };
      MemorySegment distancesTensor = prepareTensor(arena, distancesDP, distancesShape, 2, 32, 2, 2, 1);

      MemorySegment prefilter = cuvsFilter.allocate(arena);
      MemorySegment prefilterTensor;

      if (prefilterDataMemorySegment == MemorySegment.NULL) {
        cuvsFilter.type(prefilter, 0); // NO_FILTER
        cuvsFilter.addr(prefilter, 0);
      } else {
        long prefilterShape[] = { (prefilterDataLength + 31) / 32 };
        prefilterLen = prefilterShape[0];
        prefilterBytes = C_INT_BYTE_SIZE * prefilterLen;

        returnValue = cuvsRMMAlloc(cuvsResources, prefilterD, prefilterBytes);
        checkCuVSError(returnValue, "cuvsRMMAlloc");

        prefilterDP = prefilterD.get(C_POINTER, 0);

        returnValue = cudaMemcpy(prefilterDP, prefilterDataMemorySegment, prefilterBytes, 1);
        checkCudaError(returnValue, "cudaMemcpy");

        prefilterTensor = prepareTensor(arena, prefilterDP, prefilterShape, 1, 32, 1, 2, 1);

        cuvsFilter.type(prefilter, 2);
        cuvsFilter.addr(prefilter, prefilterTensor.address());
      }

      returnValue = cuvsStreamSync(cuvsResources);
      checkCuVSError(returnValue, "cuvsStreamSync");

      returnValue = cuvsBruteForceSearch(cuvsResources, bruteForceIndexReference.getMemorySegment(), queriesTensor,
          neighborsTensor, distancesTensor, prefilter);
      checkCuVSError(returnValue, "cuvsBruteForceSearch");

      returnValue = cuvsStreamSync(cuvsResources);
      checkCuVSError(returnValue, "cuvsStreamSync");

      returnValue = cudaMemcpy(neighborsMemorySegment, neighborsDP, neighborsBytes, 4);
      checkCudaError(returnValue, "cudaMemcpy");
      returnValue = cudaMemcpy(distancesMemorySegment, distancesDP, distanceBytes, 4);
      checkCudaError(returnValue, "cudaMemcpy");

      returnValue = cuvsRMMFree(cuvsResources, neighborsDP, neighborsBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsResources, distancesDP, distanceBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsResources, queriesDP, queriesBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsResources, prefilterDP, prefilterBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");

      return new BruteForceSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
          distancesMemorySegment, cuvsQuery.getTopK(), cuvsQuery.getMapping(), numQueries);
    }
  }

  @Override
  public void serialize(OutputStream outputStream) throws Throwable {
    Path path = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".bf");
    serialize(outputStream, path);
  }

  @Override
  public void serialize(OutputStream outputStream, Path tempFile) throws Throwable {
    checkNotDestroyed();
    tempFile = tempFile.toAbsolutePath();

    long cuvsRes = resources.getMemorySegment().get(cuvsResources_t, 0);
    int returnValue = cuvsBruteForceSerialize(cuvsRes, resources.getArena().allocateFrom(tempFile.toString()),
        bruteForceIndexReference.getMemorySegment());
    checkCuVSError(returnValue, "cuvsBruteForceSerialize");

    try (FileInputStream fileInputStream = new FileInputStream(tempFile.toFile())) {
      fileInputStream.transferTo(outputStream);
    } finally {
      Files.deleteIfExists(tempFile);
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
    Path tmpIndexFile = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".bf");
    tmpIndexFile = tmpIndexFile.toAbsolutePath();
    IndexReference indexReference = new IndexReference(resources);

    try (var in = inputStream; FileOutputStream fileOutputStream = new FileOutputStream(tmpIndexFile.toFile())) {
      in.transferTo(fileOutputStream);

      long cuvsRes = resources.getMemorySegment().get(cuvsResources_t, 0);
      int returnValue = cuvsBruteForceDeserialize(cuvsRes, resources.getArena().allocateFrom(tmpIndexFile.toString()),
          indexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsBruteForceDeserialize");

    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
    return indexReference;
  }

  public static BruteForceIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }
    return new Builder((CuVSResourcesImpl) cuvsResources);
  }

  /**
   * Builder helps configure and create an instance of {@link BruteForceIndex}.
   */
  public static class Builder implements BruteForceIndex.Builder {

    private float[][] dataset;
    private final CuVSResourcesImpl cuvsResources;
    private BruteForceIndexParams bruteForceIndexParams;
    private InputStream inputStream;

    /**
     * Constructs this Builder with an instance of {@link CuVSResourcesImpl}.
     *
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResourcesImpl cuvsResources) {
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
     * @param dataset a two-dimensional float array
     * @return an instance of this Builder
     */
    @Override
    public Builder withDataset(float[][] dataset) {
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
   * Holds the memory reference to a BRUTEFORCE index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResourcesImpl resources) {
      memorySegment = cuvsBruteForceIndex.allocate(resources.getArena());
    }

    /**
     * Constructs BruteForceIndexReference with an instance of MemorySegment passed
     * as a parameter.
     *
     * @param indexMemorySegment the MemorySegment instance to use for containing
     *                           index reference
     */
    protected IndexReference(MemorySegment indexMemorySegment) {
      this.memorySegment = indexMemorySegment;
    }

    /**
     * Gets the instance of index MemorySegment.
     *
     * @return index MemorySegment
     */
    protected MemorySegment getMemorySegment() {
      return memorySegment;
    }
  }
}
