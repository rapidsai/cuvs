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
import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.*;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.common.Util.concatenate;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemcpy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMAlloc;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexBuild;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexExtend;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndexSearch;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsTieredIndex_t;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.TieredIndex;
import com.nvidia.cuvs.TieredIndexParams;
import com.nvidia.cuvs.TieredIndexQuery;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraSearchParams;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import com.nvidia.cuvs.internal.panama.cuvsTieredIndexParams;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.BitSet;
import java.util.Objects;

/**
 * {@link TieredIndex} encapscaps a Tiered index, along with methods to interact
 * with it.
 * <p>
 * TieredIndex is a hybrid index that combines brute force search for small datasets
 * with ANN algorithms (like CAGRA) for larger datasets, providing optimal performance
 * across different data sizes.
 *
 * @since 25.02
 */
public class TieredIndexImpl implements TieredIndex {
  private final float[][] vectors;
  private final Dataset dataset;
  private final CuVSResourcesImpl resources;
  private final TieredIndexParams tieredIndexParameters;
  private final IndexReference tieredIndexReference;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   */
  private TieredIndexImpl(
      TieredIndexParams indexParameters,
      float[][] vectors,
      Dataset dataset,
      CuVSResourcesImpl resources)
      throws Throwable {
    this.tieredIndexParameters = indexParameters;
    this.vectors = vectors;
    this.dataset = dataset;
    this.resources = resources;
    this.tieredIndexReference = build();
    this.destroyed = false;
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   */
  private TieredIndexImpl(InputStream inputStream, CuVSResourcesImpl resources) throws Throwable {
    throw new UnsupportedOperationException("Deserialization of TieredIndex is not yet supported");
  }

  /**
   * Constructor for creating an index from an existing index reference.
   */
  private TieredIndexImpl(IndexReference indexReference, CuVSResourcesImpl resources) {
    this.vectors = null;
    this.tieredIndexParameters = null;
    this.dataset = null;
    this.resources = resources;
    this.tieredIndexReference = indexReference;
    this.destroyed = false;
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native destroy_tiered_index to de-allocate the Tiered index
   */
  @Override
  public void destroyIndex() throws Throwable {
    checkNotDestroyed();
    try {
      int returnValue = cuvsTieredIndexDestroy(tieredIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsTieredIndexDestroy");
    } finally {
      destroyed = true;
    }
  }

  /**
   * Translates C build_tiered_index function to Java
   * Invokes the native build_tiered_index function via the Panama API to build the
   * {@link TieredIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      long rows = dataset != null ? dataset.size() : vectors.length;
      long cols = dataset != null ? dataset.dimensions() : (rows > 0 ? vectors[0].length : 0);

      MemorySegment indexParamsMemorySegment =
          tieredIndexParameters != null
              ? segmentFromIndexParams(localArena, tieredIndexParameters)
              : MemorySegment.NULL;

      // Get host data
      MemorySegment hostDataSeg =
          dataset != null
              ? ((DatasetImpl) dataset).asMemorySegment()
              : Util.buildMemorySegment(localArena, vectors);

      long cuvsRes = resources.getHandle();

      // TieredIndex REQUIRES device memory - allocate it
      MemorySegment datasetD = localArena.allocate(C_POINTER);
      long datasetSize = C_FLOAT_BYTE_SIZE * rows * cols;
      int returnValue = cuvsRMMAlloc(cuvsRes, datasetD, datasetSize);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      MemorySegment datasetDP = datasetD.get(C_POINTER, 0);

      // Copy host to device
      Util.cudaMemcpy(datasetDP, hostDataSeg, datasetSize, HOST_TO_DEVICE);

      // Create tensor from device memory
      long[] datasetShape = {rows, cols};
      MemorySegment datasetTensor =
          prepareTensor(localArena, datasetDP, datasetShape, 2, 32, 2, 2, 1);

      MemorySegment index = localArena.allocate(cuvsTieredIndex_t);
      returnValue = cuvsTieredIndexCreate(index);
      checkCuVSError(returnValue, "cuvsTieredIndexCreate");

      returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      // Extract the actual index pointer that was written by Create
      MemorySegment actualIndexPtr = index.get(C_POINTER, 0);

      returnValue =
          cuvsTieredIndexBuild(cuvsRes, indexParamsMemorySegment, datasetTensor, actualIndexPtr);
      checkCuVSError(returnValue, "cuvsTieredIndexBuild");

      // Clean up device memory after build
      returnValue = cuvsRMMFree(cuvsRes, datasetDP, datasetSize);
      checkCuVSError(returnValue, "cuvsRMMFree");

      return new IndexReference(actualIndexPtr);
    }
  }

  /**
   * Translates C search_tiered_index function to Java
   * Invokes the native search_tiered_index via the Panama API for searching a
   * Tiered index.
   *
   * @param query an instance of {@link TieredIndexQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link SearchResults} containing the results
   */
  @Override
  public SearchResults search(TieredIndexQuery query) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      checkNotDestroyed();
      int topK =
          query.getMapping() != null
              ? Math.min(query.getMapping().size(), query.getTopK())
              : query.getTopK();
      long numQueries = query.getQueryVectors().length;
      long numBlocks = (long) topK * numQueries;
      int vectorDimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;

      SequenceLayout neighborsLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
      SequenceLayout distancesLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsSeg = localArena.allocate(neighborsLayout);
      MemorySegment distancesSeg = localArena.allocate(distancesLayout);

      // Get host query data
      MemorySegment hostQueriesSeg = Util.buildMemorySegment(localArena, query.getQueryVectors());

      long cuvsRes = resources.getHandle();

      // Allocate DEVICE memory for all data
      MemorySegment queriesD = localArena.allocate(C_POINTER);
      MemorySegment neighborsD = localArena.allocate(C_POINTER);
      MemorySegment distancesD = localArena.allocate(C_POINTER);

      long queriesBytes = C_FLOAT_BYTE_SIZE * numQueries * vectorDimension;
      long neighborsBytes = C_LONG_BYTE_SIZE * numQueries * topK; // 64-bit for tiered index
      long distancesBytes = C_FLOAT_BYTE_SIZE * numQueries * topK;

      int returnValue = cuvsRMMAlloc(cuvsRes, queriesD, queriesBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");
      returnValue = cuvsRMMAlloc(cuvsRes, neighborsD, neighborsBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");
      returnValue = cuvsRMMAlloc(cuvsRes, distancesD, distancesBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      // Get device pointers
      MemorySegment queriesDP = queriesD.get(C_POINTER, 0);
      MemorySegment neighborsDP = neighborsD.get(C_POINTER, 0);
      MemorySegment distancesDP = distancesD.get(C_POINTER, 0);

      // Copy queries from host to device
      returnValue =
          cudaMemcpy(queriesDP, hostQueriesSeg, queriesBytes, 1); // cudaMemcpyHostToDevice
      checkCudaError(returnValue, "cudaMemcpy");

      // Create tensors from device memory
      long queriesShape[] = {numQueries, vectorDimension};
      MemorySegment queriesTensor =
          prepareTensor(localArena, queriesDP, queriesShape, 2, 32, 2, 2, 1);
      long neighborsShape[] = {numQueries, topK};
      MemorySegment neighborsTensor =
          prepareTensor(localArena, neighborsDP, neighborsShape, 0, 64, 2, 2, 1); // 64-bit int
      long distancesShape[] = {numQueries, topK};
      MemorySegment distancesTensor =
          prepareTensor(localArena, distancesDP, distancesShape, 2, 32, 2, 2, 1);

      // Sync before prefilter setup
      returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      // Handle prefilter
      MemorySegment prefilter = cuvsFilter.allocate(localArena);
      MemorySegment prefilterD = localArena.allocate(C_POINTER);
      MemorySegment prefilterDP = MemorySegment.NULL;
      long prefilterBytes = 0;

      if (query.getPrefilter() != null) {
        BitSet[] prefilters = new BitSet[] {query.getPrefilter()};
        BitSet concatenatedFilters = concatenate(prefilters, (int) query.getNumDocs());
        long filters[] = concatenatedFilters.toLongArray();
        MemorySegment hostPrefilterSeg = buildMemorySegment(localArena, filters);

        long prefilterDataLength = query.getNumDocs() * prefilters.length;
        long prefilterShape[] = {(prefilterDataLength + 31) / 32};
        long prefilterLen = prefilterShape[0];
        prefilterBytes = C_INT_BYTE_SIZE * prefilterLen;

        // Allocate device memory for prefilter
        returnValue = cuvsRMMAlloc(cuvsRes, prefilterD, prefilterBytes);
        checkCuVSError(returnValue, "cuvsRMMAlloc");

        prefilterDP = prefilterD.get(C_POINTER, 0);

        // Copy prefilter to device
        returnValue = cudaMemcpy(prefilterDP, hostPrefilterSeg, prefilterBytes, 1);
        checkCudaError(returnValue, "cudaMemcpy");

        MemorySegment prefilterTensor =
            prepareTensor(localArena, prefilterDP, prefilterShape, 1, 32, 1, 2, 1);

        cuvsFilter.type(prefilter, 1); // BITSET
        cuvsFilter.addr(prefilter, prefilterTensor.address());
      } else {
        cuvsFilter.type(prefilter, 0); // NO_FILTER
        cuvsFilter.addr(prefilter, 0);
      }

      // Perform search
      returnValue =
          cuvsTieredIndexSearch(
              cuvsRes,
              segmentFromSearchParams(query.getCagraSearchParameters(), localArena),
              tieredIndexReference.getMemorySegment(),
              queriesTensor,
              neighborsTensor,
              distancesTensor,
              prefilter);
      checkCuVSError(returnValue, "cuvsTieredIndexSearch");

      // Copy results from device to host
      returnValue =
          cudaMemcpy(neighborsSeg, neighborsDP, neighborsBytes, 2); // cudaMemcpyDeviceToHost
      checkCudaError(returnValue, "cudaMemcpy");
      returnValue = cudaMemcpy(distancesSeg, distancesDP, distancesBytes, 2);
      checkCudaError(returnValue, "cudaMemcpy");

      // Clean up device memory
      returnValue = cuvsRMMFree(cuvsRes, queriesDP, queriesBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsRes, neighborsDP, neighborsBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsRes, distancesDP, distancesBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");

      if (prefilterDP != MemorySegment.NULL) {
        returnValue = cuvsRMMFree(cuvsRes, prefilterDP, prefilterBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
      }

      return TieredSearchResultsImpl.create(
          neighborsLayout,
          distancesLayout,
          neighborsSeg,
          distancesSeg,
          topK,
          query.getMapping(),
          numQueries);
    }
  }

  @Override
  public ExtendBuilder extend() {
    checkNotDestroyed();
    return new ExtendBuilder(this);
  }

  /**
   * Performs the actual extend operation
   */
  private void performExtend(float[][] extendVectors, Dataset extendDataset) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      long rows = extendDataset != null ? extendDataset.size() : extendVectors.length;
      long cols = extendDataset != null ? extendDataset.dimensions() : extendVectors[0].length;

      // Get host data
      MemorySegment hostDataSeg =
          extendDataset != null
              ? ((DatasetImpl) extendDataset).asMemorySegment()
              : Util.buildMemorySegment(localArena, extendVectors);

      long cuvsRes = resources.getHandle();

      // Allocate device memory for extend data
      MemorySegment datasetD = localArena.allocate(C_POINTER);
      long dataSize = C_FLOAT_BYTE_SIZE * rows * cols;
      int returnValue = cuvsRMMAlloc(cuvsRes, datasetD, dataSize);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      MemorySegment datasetDP = datasetD.get(C_POINTER, 0);

      // Copy host to device
      returnValue = cudaMemcpy(datasetDP, hostDataSeg, dataSize, 1); // cudaMemcpyHostToDevice
      checkCudaError(returnValue, "cudaMemcpy");

      // Create tensor from device memory
      long datasetShape[] = {rows, cols};
      MemorySegment datasetTensor =
          prepareTensor(localArena, datasetDP, datasetShape, 2, 32, 2, 2, 1);

      returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      returnValue =
          cuvsTieredIndexExtend(cuvsRes, datasetTensor, tieredIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsTieredIndexExtend");

      // Clean up device memory
      returnValue = cuvsRMMFree(cuvsRes, datasetDP, dataSize);
      checkCuVSError(returnValue, "cuvsRMMFree");
    }
  }

  /**
   * ExtendBuilder implementation
   */
  public static class ExtendBuilder implements TieredIndex.ExtendBuilder {
    private final TieredIndexImpl index;
    private float[][] vectors;
    private Dataset dataset;

    private ExtendBuilder(TieredIndexImpl index) {
      this.index = index;
    }

    @Override
    public ExtendBuilder withDataset(float[][] vectors) {
      this.vectors = vectors;
      return this;
    }

    @Override
    public ExtendBuilder withDataset(Dataset dataset) {
      this.dataset = dataset;
      return this;
    }

    @Override
    public void execute() throws Throwable {
      if (vectors != null && dataset != null) {
        throw new IllegalArgumentException(
            "Please specify only one type of dataset (a float[][] or a Dataset instance)");
      }
      if (vectors == null && dataset == null) {
        throw new IllegalArgumentException("Must provide vectors or dataset");
      }

      index.performExtend(vectors, dataset);
    }
  }

  /**
   * Allocates the configured index parameters in the MemorySegment.
   */
  private static MemorySegment segmentFromIndexParams(Arena arena, TieredIndexParams params) {
    MemorySegment seg = cuvsTieredIndexParams.allocate(arena);

    // Get the metric from CagraParams if available, otherwise use TieredIndex metric
    int metric;
    if (params.getCagraParams() != null) {
      // Use the metric from CagraParams to ensure consistency
      metric = params.getCagraParams().getCuvsDistanceType().value;
    } else {
      // Fallback to TieredIndex metric
      metric =
          switch (params.getMetric()) {
            case L2 -> 0;
            case INNER_PRODUCT -> 1;
            default ->
                throw new IllegalArgumentException("Unsupported metric: " + params.getMetric());
          };
    }

    cuvsTieredIndexParams.metric(seg, metric);

    int algo = 0; // CUVS_TIERED_INDEX_ALGO_CAGRA
    cuvsTieredIndexParams.algo(seg, algo);

    cuvsTieredIndexParams.min_ann_rows(seg, params.getMinAnnRows());
    cuvsTieredIndexParams.create_ann_index_on_extend(seg, params.isCreateAnnIndexOnExtend());

    CagraIndexParams cagraParams = params.getCagraParams();
    if (cagraParams != null) {
      MemorySegment cagraParamsSeg = cuvsCagraIndexParams.allocate(arena);

      cuvsCagraIndexParams.intermediate_graph_degree(
          cagraParamsSeg, cagraParams.getIntermediateGraphDegree());
      cuvsCagraIndexParams.graph_degree(cagraParamsSeg, cagraParams.getGraphDegree());
      cuvsCagraIndexParams.build_algo(cagraParamsSeg, cagraParams.getCagraGraphBuildAlgo().value);
      cuvsCagraIndexParams.nn_descent_niter(
          cagraParamsSeg, cagraParams.getNNDescentNumIterations());
      cuvsCagraIndexParams.metric(cagraParamsSeg, metric);

      cuvsTieredIndexParams.cagra_params(seg, cagraParamsSeg);
    }

    cuvsTieredIndexParams.ivf_flat_params(seg, MemorySegment.NULL);
    cuvsTieredIndexParams.ivf_pq_params(seg, MemorySegment.NULL);

    return seg;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(CagraSearchParams params, Arena arena) {
    MemorySegment seg = cuvsCagraSearchParams.allocate(arena);
    cuvsCagraSearchParams.max_queries(seg, params.getMaxQueries());
    cuvsCagraSearchParams.itopk_size(seg, params.getITopKSize());
    cuvsCagraSearchParams.max_iterations(seg, params.getMaxIterations());
    if (params.getCagraSearchAlgo() != null) {
      cuvsCagraSearchParams.algo(seg, params.getCagraSearchAlgo().value);
    }
    cuvsCagraSearchParams.team_size(seg, params.getTeamSize());
    cuvsCagraSearchParams.search_width(seg, params.getSearchWidth());
    cuvsCagraSearchParams.min_iterations(seg, params.getMinIterations());
    cuvsCagraSearchParams.thread_block_size(seg, params.getThreadBlockSize());
    if (params.getHashMapMode() != null) {
      cuvsCagraSearchParams.hashmap_mode(seg, params.getHashMapMode().value);
    }
    cuvsCagraSearchParams.hashmap_max_fill_rate(seg, params.getHashMapMaxFillRate());
    cuvsCagraSearchParams.num_random_samplings(seg, params.getNumRandomSamplings());
    cuvsCagraSearchParams.rand_xor_mask(seg, params.getRandXORMask());
    return seg;
  }

  /**
   * Gets an instance of {@link CuVSResources}
   *
   * @return an instance of {@link CuVSResources}
   */
  @Override
  public CuVSResources getCuVSResources() {
    return resources;
  }

  /**
   * Gets the index type
   *
   * @return the index type
   */
  @Override
  public TieredIndexType getIndexType() {
    TieredIndexType indexType = TieredIndexType.CAGRA; // Default to CAGRA for now
    return indexType;
  }

  /**
   * Static method to create a new builder
   */
  public static TieredIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }
    return new TieredIndexImpl.Builder((CuVSResourcesImpl) cuvsResources);
  }

  /**
   * Builder helps configure and create an instance of {@link TieredIndex}.
   */
  public static class Builder implements TieredIndex.Builder {
    private CuVSResourcesImpl resources;
    private float[][] vectors;
    private Dataset dataset;
    private TieredIndexParams params;
    private TieredIndexType indexType = TieredIndexType.CAGRA;
    private InputStream inputStream;

    private Builder(CuVSResourcesImpl resources) {
      this.resources = resources;
    }

    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    @Override
    public Builder withDataset(float[][] vectors) {
      this.vectors = vectors;
      return this;
    }

    @Override
    public Builder withDataset(Dataset dataset) {
      this.dataset = dataset;
      return this;
    }

    @Override
    public Builder withIndexParams(TieredIndexParams params) {
      this.params = params;
      return this;
    }

    @Override
    public Builder withIndexType(TieredIndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    @Override
    public TieredIndex build() throws Throwable {
      if (inputStream != null) {
        return new TieredIndexImpl(inputStream, resources);
      } else {
        if (vectors != null && dataset != null) {
          throw new IllegalArgumentException(
              "Please specify only one type of dataset (a float[][] or a Dataset instance)");
        }
        if (vectors == null && dataset == null) {
          throw new IllegalArgumentException("Must provide vectors or dataset");
        }
        if (params == null) {
          throw new IllegalStateException("Index parameters must be provided");
        }
        return new TieredIndexImpl(params, vectors, dataset, resources);
      }
    }
  }

  /**
   * Holds the memory reference to a Tiered index.
   */
  public static class IndexReference {
    private final MemorySegment memorySegment;

    /**
     * Constructs TieredIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResourcesImpl resources) {
      // Don't allocate here - the C function will allocate
      memorySegment = MemorySegment.NULL;
    }

    /**
     * Constructs TieredIndexReference with an instance of MemorySegment passed as a
     * parameter.
     */
    protected IndexReference(MemorySegment indexMemorySegment) {
      this.memorySegment = indexMemorySegment;
    }

    /**
     * Gets the instance of index MemorySegment.
     */
    protected MemorySegment getMemorySegment() {
      return memorySegment;
    }
  }
}
