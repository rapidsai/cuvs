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
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.HOST_TO_DEVICE;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.INFER_DIRECTION;
import static com.nvidia.cuvs.internal.common.Util.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.concatenate;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraBuild;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraDeserialize;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraIndexCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraIndexDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraIndex_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraMerge;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraSearch;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraSerialize;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraSerializeToHnswlib;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.omp_set_num_threads;

import com.nvidia.cuvs.CagraCompressionParams;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraMergeParams;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSIvfPqSearchParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.panama.cuvsCagraCompressionParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraMergeParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraSearchParams;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqSearchParams;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.BitSet;
import java.util.Objects;
import java.util.UUID;

/**
 * {@link CagraIndex} encapsulates a CAGRA index, along with methods to interact
 * with it.
 * <p>
 * CAGRA is a graph-based nearest neighbors algorithm that was built from the
 * ground up for GPU acceleration. CAGRA demonstrates state-of-the art index
 * build and query performance for both small and large-batch sized search. Know
 * more about this algorithm
 * <a href="https://arxiv.org/abs/2308.15136" target="_blank">here</a>
 *
 * @since 25.02
 */
public class CagraIndexImpl implements CagraIndex {
  private final CuVSResourcesImpl resources;
  private final IndexReference cagraIndexReference;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   *
   * @param indexParameters an instance of {@link CagraIndexParams} holding the
   *                        index parameters
   * @param dataset         the dataset for indexing
   * @param resources       an instance of {@link CuVSResources}
   */
  private CagraIndexImpl(
      CagraIndexParams indexParameters, Dataset dataset, CuVSResourcesImpl resources) {
    Objects.requireNonNull(dataset);
    this.resources = resources;
    assert dataset instanceof DatasetImpl;
    this.cagraIndexReference = build(indexParameters, (DatasetImpl) dataset);
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private CagraIndexImpl(InputStream inputStream, CuVSResourcesImpl resources) throws Throwable {
    this.resources = resources;
    this.cagraIndexReference = deserialize(inputStream);
  }

  /**
   * Constructor for creating an index from an existing index reference.
   * Used primarily for the merge operation.
   *
   * @param indexReference The reference to the existing index
   * @param resources The resources instance
   */
  private CagraIndexImpl(IndexReference indexReference, CuVSResourcesImpl resources) {
    this.resources = resources;
    this.cagraIndexReference = indexReference;
    this.destroyed = false;
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native destroy_cagra_index to de-allocate the CAGRA index
   */
  @Override
  public void destroyIndex() throws Throwable {
    checkNotDestroyed();
    try {
      int returnValue = cuvsCagraIndexDestroy(cagraIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsCagraIndexDestroy");
      if (cagraIndexReference.dataset != null) {
        cagraIndexReference.dataset.close();
      }
    } finally {
      destroyed = true;
    }
  }

  /**
   * Invokes the native build_cagra_index function via the Panama API to build the
   * {@link CagraIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build(CagraIndexParams indexParameters, DatasetImpl dataset) {
    try (var localArena = Arena.ofConfined()) {
      long rows = dataset.size();
      long cols = dataset.dimensions();

      MemorySegment indexParamsMemorySegment =
          indexParameters != null
              ? segmentFromIndexParams(localArena, indexParameters)
              : MemorySegment.NULL;

      int numWriterThreads = indexParameters != null ? indexParameters.getNumWriterThreads() : 1;
      omp_set_num_threads(numWriterThreads);

      MemorySegment dataSeg = dataset.asMemorySegment();

      long cuvsRes = resources.getHandle();

      long[] datasetShape = {rows, cols};
      MemorySegment datasetTensor =
          prepareTensor(localArena, dataSeg, datasetShape, 2, 32, 2, 2, 1);

      var index = createCagraIndex();

      if (cuvsCagraIndexParams.build_algo(indexParamsMemorySegment)
          == 1) { // when build algo is IVF_PQ
        MemorySegment cuvsIvfPqIndexParamsMS =
            cuvsIvfPqParams.ivf_pq_build_params(
                cuvsCagraIndexParams.graph_build_params(indexParamsMemorySegment));
        int n_lists = cuvsIvfPqIndexParams.n_lists(cuvsIvfPqIndexParamsMS);
        // As rows cannot be less than n_lists value so trim down.
        cuvsIvfPqIndexParams.n_lists(
            cuvsIvfPqIndexParamsMS, (int) (rows < n_lists ? rows : n_lists));
      }

      var returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      returnValue = cuvsCagraBuild(cuvsRes, indexParamsMemorySegment, datasetTensor, index);
      checkCuVSError(returnValue, "cuvsCagraBuild");

      returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      omp_set_num_threads(1);

      return new IndexReference(index, dataset);
    }
  }

  private static MemorySegment createCagraIndex() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexPtrPtr = localArena.allocate(cuvsCagraIndex_t);
      // cuvsCagraIndexCreate gets a pointer to a cuvsCagraIndex_t, which is defined as a pointer to
      // cuvsCagraIndex.
      // It's basically an "out" parameter: the C functions will create the index and "return back"
      // a pointer to it: (*index = new cuvsCagraIndex{};
      // The "out parameter" pointer is needed only for the duration of the function invocation (it
      // could be a stack pointer, in C) so we allocate it from our localArena
      var returnValue = cuvsCagraIndexCreate(indexPtrPtr);
      checkCuVSError(returnValue, "cuvsCagraIndexCreate");
      return indexPtrPtr.get(cuvsCagraIndex_t, 0);
    }
  }

  /**
   * Invokes the native search_cagra_index via the Panama API for searching a
   * CAGRA index.
   *
   * @param query an instance of {@link CagraQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link CagraSearchResults} containing the results
   */
  @Override
  public SearchResults search(CagraQuery query) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      checkNotDestroyed();
      int topK = query.getTopK();
      long numQueries = query.getQueryVectors().length;
      long numBlocks = topK * numQueries;
      int vectorDimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;

      SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_INT);
      SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsMemorySegment = localArena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = localArena.allocate(distancesSequenceLayout);
      MemorySegment floatsSeg = buildMemorySegment(localArena, query.getQueryVectors());

      long cuvsRes = resources.getHandle();

      long queriesBytes = C_FLOAT_BYTE_SIZE * numQueries * vectorDimension;
      long neighborsBytes = C_INT_BYTE_SIZE * numQueries * topK;
      long distancesBytes = C_FLOAT_BYTE_SIZE * numQueries * topK;
      long prefilterBytes = 0; // size assigned later

      MemorySegment queriesDP = allocateRMMSegment(cuvsRes, queriesBytes);
      MemorySegment neighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
      MemorySegment distancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
      MemorySegment prefilterDP = MemorySegment.NULL;
      long prefilterLen = 0;

      cudaMemcpy(queriesDP, floatsSeg, queriesBytes, INFER_DIRECTION);

      long[] queriesShape = {numQueries, vectorDimension};
      MemorySegment queriesTensor =
          prepareTensor(localArena, queriesDP, queriesShape, 2, 32, 2, 2, 1);
      long[] neighborsShape = {numQueries, topK};
      MemorySegment neighborsTensor =
          prepareTensor(localArena, neighborsDP, neighborsShape, 1, 32, 2, 2, 1);
      long[] distancesShape = {numQueries, topK};
      MemorySegment distancesTensor =
          prepareTensor(localArena, distancesDP, distancesShape, 2, 32, 2, 2, 1);

      var returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      // prepare the prefiltering data
      long prefilterDataLength = 0;
      MemorySegment prefilterDataMemorySegment = MemorySegment.NULL;
      BitSet[] prefilters;
      if (query.getPrefilter() != null) {
        prefilters = new BitSet[] {query.getPrefilter()};
        BitSet concatenatedFilters = concatenate(prefilters, query.getNumDocs());
        long[] filters = concatenatedFilters.toLongArray();
        prefilterDataMemorySegment = buildMemorySegment(localArena, filters);
        prefilterDataLength = query.getNumDocs() * prefilters.length;
      }

      MemorySegment prefilter = cuvsFilter.allocate(localArena);
      MemorySegment prefilterTensor;

      if (prefilterDataMemorySegment == MemorySegment.NULL) {
        cuvsFilter.type(prefilter, 0); // NO_FILTER
        cuvsFilter.addr(prefilter, 0);
      } else {
        long[] prefilterShape = {(prefilterDataLength + 31) / 32};
        prefilterLen = prefilterShape[0];
        prefilterBytes = C_INT_BYTE_SIZE * prefilterLen;

        prefilterDP = allocateRMMSegment(cuvsRes, prefilterBytes);

        cudaMemcpy(prefilterDP, prefilterDataMemorySegment, prefilterBytes, HOST_TO_DEVICE);

        prefilterTensor = prepareTensor(localArena, prefilterDP, prefilterShape, 1, 32, 1, 2, 1);

        cuvsFilter.type(prefilter, 1);
        cuvsFilter.addr(prefilter, prefilterTensor.address());
      }

      returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      returnValue =
          cuvsCagraSearch(
              cuvsRes,
              segmentFromSearchParams(localArena, query.getCagraSearchParameters()),
              cagraIndexReference.getMemorySegment(),
              queriesTensor,
              neighborsTensor,
              distancesTensor,
              prefilter);
      checkCuVSError(returnValue, "cuvsCagraSearch");

      returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      cudaMemcpy(neighborsMemorySegment, neighborsDP, neighborsBytes, INFER_DIRECTION);
      cudaMemcpy(distancesMemorySegment, distancesDP, distancesBytes, INFER_DIRECTION);

      returnValue = cuvsRMMFree(cuvsRes, distancesDP, distancesBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsRes, neighborsDP, neighborsBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsRes, queriesDP, queriesBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");

      if (prefilterLen > 0) {
        returnValue = cuvsRMMFree(cuvsRes, prefilterDP, C_INT_BYTE_SIZE * prefilterBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
      }

      return CagraSearchResults.create(
          neighborsSequenceLayout,
          distancesSequenceLayout,
          neighborsMemorySegment,
          distancesMemorySegment,
          topK,
          query.getMapping(),
          numQueries);
    }
  }

  @Override
  public void serialize(OutputStream outputStream) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag");
    serialize(outputStream, path, 1024);
  }

  @Override
  public void serialize(OutputStream outputStream, int bufferLength) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag");
    serialize(outputStream, path, bufferLength);
  }

  @Override
  public void serialize(OutputStream outputStream, Path tempFile, int bufferLength)
      throws Throwable {
    checkNotDestroyed();
    tempFile = tempFile.toAbsolutePath();
    try (var localArena = Arena.ofConfined()) {

      long cuvsRes = resources.getHandle();
      var returnValue =
          cuvsCagraSerialize(
              cuvsRes,
              localArena.allocateFrom(tempFile.toString()),
              cagraIndexReference.getMemorySegment(),
              true);
      checkCuVSError(returnValue, "cuvsCagraSerialize");

      try (var fileInputStream = Files.newInputStream(tempFile)) {
        byte[] chunk = new byte[bufferLength];
        int chunkLength = 0;
        while ((chunkLength = fileInputStream.read(chunk)) != -1) {
          outputStream.write(chunk, 0, chunkLength);
        }
      } finally {
        Files.deleteIfExists(tempFile);
      }
    }
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
    serializeToHNSW(outputStream, path, 1024);
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
    serializeToHNSW(outputStream, path, bufferLength);
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream, Path tempFile, int bufferLength)
      throws Throwable {
    checkNotDestroyed();
    tempFile = tempFile.toAbsolutePath();

    try (var localArena = Arena.ofConfined()) {
      MemorySegment pathSeg = buildMemorySegment(localArena, tempFile.toString());

      long cuvsRes = resources.getHandle();
      int returnValue =
          cuvsCagraSerializeToHnswlib(cuvsRes, pathSeg, cagraIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsCagraSerializeToHnswlib");
    }

    try (FileInputStream fileInputStream = new FileInputStream(tempFile.toFile())) {
      byte[] chunk = new byte[bufferLength];
      int chunkLength;
      while ((chunkLength = fileInputStream.read(chunk)) != -1) {
        outputStream.write(chunk, 0, chunkLength);
      }
    } finally {
      Files.deleteIfExists(tempFile);
    }
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a CAGRA index
   * using an {@link InputStream}.
   *
   * @param inputStream  an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    Path tmpIndexFile =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag")
            .toAbsolutePath();
    var index = createCagraIndex();

    try (inputStream;
        var outputStream = Files.newOutputStream(tmpIndexFile);
        var arena = Arena.ofConfined()) {
      inputStream.transferTo(outputStream);

      long cuvsRes = resources.getHandle();
      var returnValue =
          cuvsCagraDeserialize(cuvsRes, arena.allocateFrom(tmpIndexFile.toString()), index);
      checkCuVSError(returnValue, "cuvsCagraDeserialize");

    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
    return new IndexReference(index, null);
  }

  /**
   * Gets an instance of {@link CuVSResources}
   *
   * @return an instance of {@link CuVSResources}
   */
  @Override
  public CuVSResourcesImpl getCuVSResources() {
    return resources;
  }

  /**
   * Allocates the configured index parameters in the MemorySegment.
   */
  private static MemorySegment segmentFromIndexParams(Arena arena, CagraIndexParams params) {
    MemorySegment seg = cuvsCagraIndexParams.allocate(arena);
    cuvsCagraIndexParams.intermediate_graph_degree(seg, params.getIntermediateGraphDegree());
    cuvsCagraIndexParams.graph_degree(seg, params.getGraphDegree());
    cuvsCagraIndexParams.build_algo(seg, params.getCagraGraphBuildAlgo().value);
    cuvsCagraIndexParams.nn_descent_niter(seg, params.getNNDescentNumIterations());
    cuvsCagraIndexParams.metric(seg, params.getCuvsDistanceType().value);

    CagraCompressionParams cagraCompressionParams = params.getCagraCompressionParams();
    if (cagraCompressionParams != null) {
      MemorySegment cuvsCagraCompressionParamsMemorySegment =
          cuvsCagraCompressionParams.allocate(arena);
      cuvsCagraCompressionParams.pq_bits(
          cuvsCagraCompressionParamsMemorySegment, cagraCompressionParams.getPqBits());
      cuvsCagraCompressionParams.pq_dim(
          cuvsCagraCompressionParamsMemorySegment, cagraCompressionParams.getPqDim());
      cuvsCagraCompressionParams.vq_n_centers(
          cuvsCagraCompressionParamsMemorySegment, cagraCompressionParams.getVqNCenters());
      cuvsCagraCompressionParams.kmeans_n_iters(
          cuvsCagraCompressionParamsMemorySegment, cagraCompressionParams.getKmeansNIters());
      cuvsCagraCompressionParams.vq_kmeans_trainset_fraction(
          cuvsCagraCompressionParamsMemorySegment,
          cagraCompressionParams.getVqKmeansTrainsetFraction());
      cuvsCagraCompressionParams.pq_kmeans_trainset_fraction(
          cuvsCagraCompressionParamsMemorySegment,
          cagraCompressionParams.getPqKmeansTrainsetFraction());
      cuvsCagraIndexParams.compression(seg, cuvsCagraCompressionParamsMemorySegment);
    }

    if (params.getCagraGraphBuildAlgo().equals(CagraGraphBuildAlgo.IVF_PQ)) {

      MemorySegment ivfpqIndexParamsMemorySegment = cuvsIvfPqIndexParams.allocate(arena);
      CuVSIvfPqIndexParams cuVSIvfPqIndexParams = params.getCuVSIvfPqParams().getIndexParams();

      cuvsIvfPqIndexParams.metric(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getMetric().value);
      cuvsIvfPqIndexParams.metric_arg(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getMetricArg());
      cuvsIvfPqIndexParams.add_data_on_build(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.isAddDataOnBuild());
      cuvsIvfPqIndexParams.n_lists(ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getnLists());
      cuvsIvfPqIndexParams.kmeans_n_iters(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getKmeansNIters());
      cuvsIvfPqIndexParams.kmeans_trainset_fraction(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getKmeansTrainsetFraction());
      cuvsIvfPqIndexParams.pq_bits(ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getPqBits());
      cuvsIvfPqIndexParams.pq_dim(ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getPqDim());
      cuvsIvfPqIndexParams.codebook_kind(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getCodebookKind().value);
      cuvsIvfPqIndexParams.force_random_rotation(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.isForceRandomRotation());
      cuvsIvfPqIndexParams.conservative_memory_allocation(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.isConservativeMemoryAllocation());
      cuvsIvfPqIndexParams.max_train_points_per_pq_code(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getMaxTrainPointsPerPqCode());

      MemorySegment ivfpqSearchParamsMemorySegment = cuvsIvfPqSearchParams.allocate(arena);
      CuVSIvfPqSearchParams cuVSIvfPqSearchParams = params.getCuVSIvfPqParams().getSearchParams();
      cuvsIvfPqSearchParams.n_probes(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getnProbes());
      cuvsIvfPqSearchParams.lut_dtype(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getLutDtype().value);
      cuvsIvfPqSearchParams.internal_distance_dtype(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getInternalDistanceDtype().value);
      cuvsIvfPqSearchParams.preferred_shmem_carveout(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getPreferredShmemCarveout());

      MemorySegment cuvsIvfPqParamsMemorySegment = cuvsIvfPqParams.allocate(arena);
      cuvsIvfPqParams.ivf_pq_build_params(
          cuvsIvfPqParamsMemorySegment, ivfpqIndexParamsMemorySegment);
      cuvsIvfPqParams.ivf_pq_search_params(
          cuvsIvfPqParamsMemorySegment, ivfpqSearchParamsMemorySegment);
      cuvsIvfPqParams.refinement_rate(
          cuvsIvfPqParamsMemorySegment, params.getCuVSIvfPqParams().getRefinementRate());

      cuvsCagraIndexParams.graph_build_params(seg, cuvsIvfPqParamsMemorySegment);
    }

    return seg;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(Arena arena, CagraSearchParams params) {
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

  public static CagraIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }
    return new CagraIndexImpl.Builder((CuVSResourcesImpl) cuvsResources);
  }

  /**
   * Merges multiple CAGRA indexes into a single index.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @return A new merged CAGRA index
   */
  public static CagraIndex merge(CagraIndex[] indexes) {
    return merge(indexes, null);
  }

  /**
   * Merges multiple CAGRA indexes into a single index with specified merge parameters.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   */
  public static CagraIndex merge(CagraIndex[] indexes, CagraMergeParams mergeParams) {
    CuVSResourcesImpl resources = (CuVSResourcesImpl) indexes[0].getCuVSResources();
    var mergedIndex = createCagraIndex();
    long cuvsRes = resources.getHandle();

    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexesSegment =
          localArena.allocate(indexes.length * ValueLayout.ADDRESS.byteSize());

      for (int i = 0; i < indexes.length; i++) {
        CagraIndexImpl indexImpl = (CagraIndexImpl) indexes[i];
        indexesSegment.setAtIndex(
            ValueLayout.ADDRESS, i, indexImpl.cagraIndexReference.getMemorySegment());
      }

      // TODO: we should call cuvsCreateMergeParams here, instead of allocating this ourselves
      // See https://github.com/rapidsai/cuvs/pull/1109
      var mergeParamsSegment = createMergeParamsSegment(localArena, mergeParams);

      var returnValue =
          cuvsCagraMerge(cuvsRes, mergeParamsSegment, indexesSegment, indexes.length, mergedIndex);

      checkCuVSError(returnValue, "cuvsCagraMerge");
    }

    return new CagraIndexImpl(new IndexReference(mergedIndex, null), resources);
  }

  /**
   * Creates (allocates and fill) native memory version of merge parameters.
   *
   * @param arena       The arena to use to allocate the MemorySegment(s) that will hold the merge parameters data
   * @param mergeParams The merge parameters
   * @return A memory segment pointing to the native structure for merge parameters
   */
  private static MemorySegment createMergeParamsSegment(Arena arena, CagraMergeParams mergeParams) {
    MemorySegment seg = cuvsCagraMergeParams.allocate(arena);

    if (mergeParams != null) {
      if (mergeParams.getOutputIndexParams() != null) {
        MemorySegment outputIndexParamsSeg =
            segmentFromIndexParams(arena, mergeParams.getOutputIndexParams());
        cuvsCagraMergeParams.output_index_params(seg, outputIndexParamsSeg);
      } else {
        cuvsCagraMergeParams.output_index_params(seg, MemorySegment.NULL);
      }

      cuvsCagraMergeParams.strategy(seg, mergeParams.getStrategy().value);
    } else {
      MemorySegment outputIndexParamsSeg =
          segmentFromIndexParams(arena, new CagraIndexParams.Builder().build());
      cuvsCagraMergeParams.output_index_params(seg, outputIndexParamsSeg);
    }

    return seg;
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  public static class Builder implements CagraIndex.Builder {

    private Dataset dataset;
    private CagraIndexParams cagraIndexParams;
    private final CuVSResourcesImpl cuvsResources;
    private InputStream inputStream;

    public Builder(CuVSResourcesImpl cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    @Override
    public Builder withDataset(float[][] vectors) {
      this.dataset = Dataset.ofArray(vectors);
      return this;
    }

    @Override
    public Builder withDataset(Dataset dataset) {
      this.dataset = dataset;
      return this;
    }

    @Override
    public Builder withIndexParams(CagraIndexParams cagraIndexParameters) {
      this.cagraIndexParams = cagraIndexParameters;
      return this;
    }

    @Override
    public CagraIndexImpl build() throws Throwable {
      if (inputStream != null) {
        return new CagraIndexImpl(inputStream, cuvsResources);
      } else {
        return new CagraIndexImpl(cagraIndexParams, dataset, cuvsResources);
      }
    }
  }

  /**
   * Holds the memory reference to a CAGRA index.
   */
  public static class IndexReference {

    private final MemorySegment memorySegment;
    private final Dataset dataset;

    /**
     * Constructs CagraIndexReference with an instance of MemorySegment passed as a
     * parameter.
     *
     * @param indexMemorySegment the MemorySegment instance to use for containing
     *                           index reference
     * @param dataset            the dataset used for indexing; the dataset lifetime
     *                           matches the lifetime of the index, we need to keep a reference
     *                           to it so we can close it when the index is closed.
     *                           Can be null (e.g. from deserialization or merging)
     */
    private IndexReference(MemorySegment indexMemorySegment, Dataset dataset) {
      this.memorySegment = indexMemorySegment;
      this.dataset = dataset;
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
