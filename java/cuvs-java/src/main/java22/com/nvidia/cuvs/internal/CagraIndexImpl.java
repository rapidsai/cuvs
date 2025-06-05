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
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.internal.common.Util.checkError;
import static java.lang.foreign.ValueLayout.ADDRESS;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

import com.nvidia.cuvs.CagraCompressionParams;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraMergeParams;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.cuvsCagraCompressionParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndex;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraMergeParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraSearchParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqSearchParams;
import java.util.BitSet;

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

  private static final MethodHandle indexMethodHandle = downcallHandle("build_cagra_index",
      FunctionDescriptor.of(ADDRESS, ADDRESS, C_LONG, C_LONG, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_INT));

  private static final MethodHandle searchMethodHandle = downcallHandle("search_cagra_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, C_INT, C_LONG, C_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_LONG));

  private static final MethodHandle serializeMethodHandle = downcallHandle("serialize_cagra_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle deserializeMethodHandle = downcallHandle("deserialize_cagra_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle destroyIndexMethodHandle = downcallHandle("destroy_cagra_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

  private static final MethodHandle serializeCAGRAIndexToHNSWMethodHandle = downcallHandle("serialize_cagra_index_to_hnsw",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));


  private static final MethodHandle mergeMethodHandle = downcallHandle("merge_cagra_indexes",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, C_INT, ADDRESS, ADDRESS));

  private final float[][] vectors;
  private final Dataset dataset;
  private final CuVSResourcesImpl resources;
  private final CagraIndexParams cagraIndexParameters;
  private final CagraCompressionParams cagraCompressionParams;
  private final IndexReference cagraIndexReference;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   *
   * @param indexParameters        an instance of {@link CagraIndexParams} holding
   *                               the index parameters
   * @param cagraCompressionParams an instance of {@link CagraCompressionParams}
   *                               holding the compression parameters
   * @param dataset                the dataset for indexing
   * @param resources              an instance of {@link CuVSResources}
   */
  private CagraIndexImpl(CagraIndexParams indexParameters, CagraCompressionParams cagraCompressionParams, float[][] vectors,
      Dataset dataset, CuVSResourcesImpl resources) throws Throwable {
    this.cagraIndexParameters = indexParameters;
    this.cagraCompressionParams = cagraCompressionParams;
    this.vectors = vectors;
    this.dataset = dataset;
    this.resources = resources;
    this.cagraIndexReference = build();
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private CagraIndexImpl(InputStream inputStream, CuVSResourcesImpl resources) throws Throwable {
    this.cagraIndexParameters = null;
    this.cagraCompressionParams = null;
    this.vectors = null;
    this.dataset = null;
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
    this.vectors = null;
    this.cagraIndexParameters = null;
    this.cagraCompressionParams = null;
    this.dataset = null;
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
    try (var arena = Arena.ofConfined()) {
      MemorySegment returnValue = arena.allocate(C_INT);
      destroyIndexMethodHandle.invokeExact(cagraIndexReference.getMemorySegment(), returnValue);
      checkError(returnValue.get(C_INT, 0L), "destroyIndexMethodHandle");
    } finally {
      destroyed = true;
    }
    if (dataset != null) dataset.close();
  }

  /**
   * Invokes the native build_cagra_index function via the Panama API to build the
   * {@link CagraIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    long rows = dataset != null? dataset.size(): vectors.length;
    long cols = dataset != null? dataset.dimensions(): (rows > 0 ? vectors[0].length : 0);

    MemorySegment indexParamsMemorySegment = cagraIndexParameters != null
        ? segmentFromIndexParams(resources, cagraIndexParameters)
        : MemorySegment.NULL;

    int numWriterThreads = cagraIndexParameters != null ? cagraIndexParameters.getNumWriterThreads() : 1;

    MemorySegment compressionParamsMemorySegment = cagraCompressionParams != null
        ? segmentFromCompressionParams(cagraCompressionParams)
        : MemorySegment.NULL;

    MemorySegment dataSeg = dataset != null? ((DatasetImpl) dataset).seg:
    	Util.buildMemorySegment(resources.getArena(), vectors);

    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      var indexSeg = (MemorySegment) indexMethodHandle.invokeExact(
        dataSeg,
        rows,
        cols,
        resources.getMemorySegment(),
        returnValue,
        indexParamsMemorySegment,
        compressionParamsMemorySegment,
        numWriterThreads
      );
      checkError(returnValue.get(C_INT, 0L), "indexMethodHandle");
      return new IndexReference(indexSeg);
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
    checkNotDestroyed();
    int topK = query.getMapping() != null ? Math.min(query.getMapping().size(), query.getTopK()) : query.getTopK();
    long numQueries = query.getQueryVectors().length;
    long numBlocks = topK * numQueries;
    int vectorDimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_INT);
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
    MemorySegment neighborsMemorySegment = resources.getArena().allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.getArena().allocate(distancesSequenceLayout);
    MemorySegment floatsSeg = Util.buildMemorySegment(resources.getArena(), query.getQueryVectors());

    long prefilterDataLength = 0;
    MemorySegment prefilterData = MemorySegment.NULL;
    if (query.getPrefilter() != null) {
      long[] longArray = query.getPrefilter().toLongArray();
      prefilterData = Util.buildMemorySegment(resources.getArena(), longArray);
      prefilterDataLength = query.getNumDocs();
    }


    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      searchMethodHandle.invokeExact(
        cagraIndexReference.getMemorySegment(),
        floatsSeg,
        topK,
        numQueries,
        vectorDimension,
        resources.getMemorySegment(),
        neighborsMemorySegment,
        distancesMemorySegment,
        returnValue,
        segmentFromSearchParams(query.getCagraSearchParameters()),
        prefilterData,
        prefilterDataLength
      );
      checkError(returnValue.get(C_INT, 0L), "searchMethodHandle");
    }
    return new CagraSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, topK, query.getMapping(), numQueries);
  }

  @Override
  public void serialize(OutputStream outputStream) throws Throwable {
    Path p = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag");
    serialize(outputStream, p, 1024);
  }

  @Override
  public void serialize(OutputStream outputStream, int bufferLength) throws Throwable {
    Path p = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag");
    serialize(outputStream, p, bufferLength);
  }

  @Override
  public void serialize(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable {
    checkNotDestroyed();
    tempFile = tempFile.toAbsolutePath();
    MemorySegment pathSeg = Util.buildMemorySegment(resources.getArena(), tempFile.toString());
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      serializeMethodHandle.invokeExact(
        resources.getMemorySegment(),
        cagraIndexReference.getMemorySegment(),
        returnValue,
        pathSeg
      );
      checkError(returnValue.get(C_INT, 0L), "serializeMethodHandle");

      try (FileInputStream fileInputStream = new FileInputStream(tempFile.toFile())) {
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
    Path p = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
    serializeToHNSW(outputStream, p, 1024);
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable {
    Path p = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
    serializeToHNSW(outputStream, p, bufferLength);
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable {
    checkNotDestroyed();
    tempFile = tempFile.toAbsolutePath();
    MemorySegment pathSeg = Util.buildMemorySegment(resources.getArena(), tempFile.toString());
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      serializeCAGRAIndexToHNSWMethodHandle.invokeExact(
        resources.getMemorySegment(),
        pathSeg,
        cagraIndexReference.getMemorySegment(),
        returnValue
      );
      checkError(returnValue.get(C_INT, 0L), "serializeCAGRAIndexToHNSWMethodHandle");

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
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a CAGRA index
   * using an {@link InputStream}.
   *
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    return deserialize(inputStream, 1024);
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a CAGRA index
   * using an {@link InputStream}.
   *
   * @param inputStream  an instance of {@link InputStream}
   * @param bufferLength the length of the buffer to use while reading the bytes
   *                     from the stream. Default value is 1024.
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream, int bufferLength) throws Throwable {
    Path tmpIndexFile = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag");
    tmpIndexFile = tmpIndexFile.toAbsolutePath();
    IndexReference indexReference = new IndexReference(resources);

    try (var in = inputStream;
         FileOutputStream fileOutputStream = new FileOutputStream(tmpIndexFile.toFile())) {
      in.transferTo(fileOutputStream);
      MemorySegment pathSeg = Util.buildMemorySegment(resources.getArena(), tmpIndexFile.toString());
      try (var localArena = Arena.ofConfined()) {
        MemorySegment returnValue = localArena.allocate(C_INT);
        deserializeMethodHandle.invokeExact(
          resources.getMemorySegment(),
          indexReference.getMemorySegment(),
          returnValue,
          pathSeg
        );
        checkError(returnValue.get(C_INT, 0L), "deserializeMethodHandle");
      }
    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
    return indexReference;
  }

  /**
   * Gets an instance of {@link CagraIndexParams}
   *
   * @return an instance of {@link CagraIndexParams}
   */
  @Override
  public CagraIndexParams getCagraIndexParameters() {
    return cagraIndexParameters;
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
   * Allocates the configured compression parameters in the MemorySegment.
   */
  private MemorySegment segmentFromCompressionParams(CagraCompressionParams params) {
    MemorySegment seg = cuvsCagraCompressionParams.allocate(resources.getArena());
    cuvsCagraCompressionParams.pq_bits(seg, params.getPqBits());
    cuvsCagraCompressionParams.pq_dim(seg, params.getPqDim());
    cuvsCagraCompressionParams.vq_n_centers(seg, params.getVqNCenters());
    cuvsCagraCompressionParams.kmeans_n_iters(seg, params.getKmeansNIters());
    cuvsCagraCompressionParams.vq_kmeans_trainset_fraction(seg, params.getVqKmeansTrainsetFraction());
    cuvsCagraCompressionParams.pq_kmeans_trainset_fraction(seg, params.getPqKmeansTrainsetFraction());
    return seg;
  }

  /**
   * Allocates the configured index parameters in the MemorySegment.
   */
  private static MemorySegment segmentFromIndexParams(CuVSResourcesImpl resources, CagraIndexParams params) {
    MemorySegment seg = cuvsCagraIndexParams.allocate(resources.getArena());
    cuvsCagraIndexParams.intermediate_graph_degree(seg, params.getIntermediateGraphDegree());
    cuvsCagraIndexParams.graph_degree(seg, params.getGraphDegree());
    cuvsCagraIndexParams.build_algo(seg, params.getCagraGraphBuildAlgo().value);
    cuvsCagraIndexParams.nn_descent_niter(seg, params.getNNDescentNumIterations());
    cuvsCagraIndexParams.metric(seg, params.getCuvsDistanceType().value);


    if (params.getCagraGraphBuildAlgo().equals(CagraGraphBuildAlgo.IVF_PQ)) {

      MemorySegment ivfpqIndexParamsMemorySegment = cuvsIvfPqIndexParams.allocate(resources.getArena());
      cuvsIvfPqIndexParams.metric(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getMetric().value);
      cuvsIvfPqIndexParams.metric_arg(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getMetricArg());
      cuvsIvfPqIndexParams.add_data_on_build(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().isAddDataOnBuild());
      cuvsIvfPqIndexParams.n_lists(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getnLists());
      cuvsIvfPqIndexParams.kmeans_n_iters(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getKmeansNIters());
      cuvsIvfPqIndexParams.kmeans_trainset_fraction(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getKmeansTrainsetFraction());
      cuvsIvfPqIndexParams.pq_bits(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getPqBits());
      cuvsIvfPqIndexParams.pq_dim(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getPqDim());
      cuvsIvfPqIndexParams.codebook_kind(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getCodebookKind().value);
      cuvsIvfPqIndexParams.force_random_rotation(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().isForceRandomRotation());
      cuvsIvfPqIndexParams.conservative_memory_allocation(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().isConservativeMemoryAllocation());
      cuvsIvfPqIndexParams.max_train_points_per_pq_code(ivfpqIndexParamsMemorySegment, params.getCuVSIvfPqParams().getIndexParams().getMaxTrainPointsPerPqCode());

      MemorySegment ivfpqSearchParamsMemorySegment = cuvsIvfPqSearchParams.allocate(resources.getArena());
      cuvsIvfPqSearchParams.n_probes(ivfpqSearchParamsMemorySegment, params.getCuVSIvfPqParams().getSearchParams().getnProbes());
      cuvsIvfPqSearchParams.lut_dtype(ivfpqSearchParamsMemorySegment, params.getCuVSIvfPqParams().getSearchParams().getLutDtype().value);
      cuvsIvfPqSearchParams.internal_distance_dtype(ivfpqSearchParamsMemorySegment, params.getCuVSIvfPqParams().getSearchParams().getInternalDistanceDtype().value);
      cuvsIvfPqSearchParams.preferred_shmem_carveout(ivfpqSearchParamsMemorySegment, params.getCuVSIvfPqParams().getSearchParams().getPreferredShmemCarveout());

      MemorySegment cuvsIvfPqParamsMemorySegment = cuvsIvfPqParams.allocate(resources.getArena());
      cuvsIvfPqParams.ivf_pq_build_params(cuvsIvfPqParamsMemorySegment, ivfpqIndexParamsMemorySegment);
      cuvsIvfPqParams.ivf_pq_search_params(cuvsIvfPqParamsMemorySegment, ivfpqSearchParamsMemorySegment);
      cuvsIvfPqParams.refinement_rate(cuvsIvfPqParamsMemorySegment, params.getCuVSIvfPqParams().getRefinementRate());

      cuvsCagraIndexParams.graph_build_params(seg, cuvsIvfPqParamsMemorySegment);
    }

    return seg;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(CagraSearchParams params) {
    MemorySegment seg = cuvsCagraSearchParams.allocate(resources.getArena());
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
    return new CagraIndexImpl.Builder((CuVSResourcesImpl)cuvsResources);
  }

  /**
   * Merges multiple CAGRA indexes into a single index.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  public static CagraIndex merge(CagraIndex[] indexes) throws Throwable {
    return merge(indexes, null);
  }

  /**
   * Merges multiple CAGRA indexes into a single index with specified merge parameters.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  public static CagraIndex merge(CagraIndex[] indexes, CagraMergeParams mergeParams) throws Throwable {
    CuVSResourcesImpl resources = (CuVSResourcesImpl) indexes[0].getCuVSResources();
    IndexReference mergedIndexReference = new IndexReference(resources);

    try (var arena = Arena.ofConfined()) {
      MemorySegment indexesSegment = arena.allocate(indexes.length * ADDRESS.byteSize());
      for (int i = 0; i < indexes.length; i++) {
        CagraIndexImpl indexImpl = (CagraIndexImpl) indexes[i];
        indexesSegment.setAtIndex(ADDRESS, i, indexImpl.cagraIndexReference.getMemorySegment());
      }

      MemorySegment returnValue = arena.allocate(C_INT);

      MemorySegment mergeParamsSegment = MemorySegment.NULL;
      if (mergeParams != null) {
        mergeParamsSegment = createMergeParamsSegment(mergeParams, resources);
      }

      mergeMethodHandle.invokeExact(
        resources.getMemorySegment(),
        indexesSegment,
        mergedIndexReference.getMemorySegment(),
        indexes.length,
        returnValue,
        mergeParamsSegment
      );

      checkError(returnValue.get(C_INT, 0L), "mergeMethodHandle");
    }

    return new CagraIndexImpl(mergedIndexReference, resources);
  }

  /**
   * Creates a memory segment for merge parameters.
   *
   * @param mergeParams The merge parameters
   * @param resources The CuVS resources
   * @return A memory segment with the merge parameters
   */
  private static MemorySegment createMergeParamsSegment(CagraMergeParams mergeParams, CuVSResourcesImpl resources) {
    MemorySegment seg = cuvsCagraMergeParams.allocate(resources.getArena());

    if (mergeParams.getOutputIndexParams() != null) {
      MemorySegment outputIndexParamsSeg = segmentFromIndexParams(resources, mergeParams.getOutputIndexParams());
      cuvsCagraMergeParams.output_index_params(seg, outputIndexParamsSeg);
    } else {
      cuvsCagraMergeParams.output_index_params(seg, MemorySegment.NULL);
    }

    cuvsCagraMergeParams.strategy(seg, mergeParams.getStrategy().value);

    return seg;
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  public static class Builder implements CagraIndex.Builder{

    private float[][] vectors;
    private Dataset dataset;
    private CagraIndexParams cagraIndexParams;
    private CagraCompressionParams cagraCompressionParams;
    private CuVSResourcesImpl cuvsResources;
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
      this.vectors = vectors;
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
    public Builder withCompressionParams(CagraCompressionParams cagraCompressionParams) {
      this.cagraCompressionParams = cagraCompressionParams;
      return this;
    }

    @Override
    public CagraIndexImpl build() throws Throwable {
      if (inputStream != null) {
        return new CagraIndexImpl(inputStream, cuvsResources);
      } else {
    	if (vectors != null && dataset != null) {
    		throw new IllegalArgumentException("Please specify only one type of dataset (a float[] or a Dataset instance)");
    	}
        return new CagraIndexImpl(cagraIndexParams, cagraCompressionParams, vectors, dataset, cuvsResources);
      }
    }
  }

  /**
   * Holds the memory reference to a CAGRA index.
   */
  public static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResourcesImpl resources) {
      memorySegment = cuvsCagraIndex.allocate(resources.getArena());
    }

    /**
     * Constructs CagraIndexReference with an instance of MemorySegment passed as a
     * parameter.
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
