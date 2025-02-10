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
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.CuVSCagraCompressionParams;
import com.nvidia.cuvs.internal.panama.CuVSCagraIndex;
import com.nvidia.cuvs.internal.panama.CuVSCagraIndexParams;
import com.nvidia.cuvs.internal.panama.CuVSCagraSearchParams;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.internal.common.Util.checkError;
import static java.lang.foreign.ValueLayout.ADDRESS;

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
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, C_INT, C_LONG, C_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle serializeMethodHandle = downcallHandle("serialize_cagra_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle deserializeMethodHandle = downcallHandle("deserialize_cagra_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle destroyIndexMethodHandle = downcallHandle("destroy_cagra_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

  private static final MethodHandle serializeCAGRAIndexToHNSWMethodHandle = downcallHandle("serialize_cagra_index_to_hnsw",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private final float[][] dataset;
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
  private CagraIndexImpl(CagraIndexParams indexParameters, CagraCompressionParams cagraCompressionParams, float[][] dataset,
      CuVSResourcesImpl resources) throws Throwable {
    this.cagraIndexParameters = indexParameters;
    this.cagraCompressionParams = cagraCompressionParams;
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
    this.dataset = null;
    this.resources = resources;
    this.cagraIndexReference = deserialize(inputStream);
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
  }

  /**
   * Invokes the native build_cagra_index function via the Panama API to build the
   * {@link CagraIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = rows > 0 ? dataset[0].length : 0;

    MemorySegment indexParamsMemorySegment = cagraIndexParameters != null
        ? segmentFromIndexParams(cagraIndexParameters)
        : MemorySegment.NULL;

    int numWriterThreads = cagraIndexParameters != null ? cagraIndexParameters.getNumWriterThreads() : 1;

    MemorySegment compressionParamsMemorySegment = cagraCompressionParams != null
        ? segmentFromCompressionParams(cagraCompressionParams)
        : MemorySegment.NULL;

    MemorySegment dataSeg = Util.buildMemorySegment(resources.getArena(), dataset);

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
        segmentFromSearchParams(query.getCagraSearchParameters())
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
    MemorySegment seg = CuVSCagraCompressionParams.allocate(resources.getArena());
    CuVSCagraCompressionParams.pq_bits(seg, params.getPqBits());
    CuVSCagraCompressionParams.pq_dim(seg, params.getPqDim());
    CuVSCagraCompressionParams.vq_n_centers(seg, params.getVqNCenters());
    CuVSCagraCompressionParams.kmeans_n_iters(seg, params.getKmeansNIters());
    CuVSCagraCompressionParams.vq_kmeans_trainset_fraction(seg, params.getVqKmeansTrainsetFraction());
    CuVSCagraCompressionParams.pq_kmeans_trainset_fraction(seg, params.getPqKmeansTrainsetFraction());
    return seg;
  }

  /**
   * Allocates the configured index parameters in the MemorySegment.
   */
  private MemorySegment segmentFromIndexParams(CagraIndexParams params) {
    MemorySegment seg = CuVSCagraIndexParams.allocate(resources.getArena());
    CuVSCagraIndexParams.intermediate_graph_degree(seg, params.getIntermediateGraphDegree());
    CuVSCagraIndexParams.graph_degree(seg, params.getGraphDegree());
    CuVSCagraIndexParams.build_algo(seg, params.getCagraGraphBuildAlgo().value);
    CuVSCagraIndexParams.nn_descent_niter(seg, params.getNNDescentNumIterations());
    CuVSCagraIndexParams.metric(seg, params.getCuvsDistanceType().value);
    return seg;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(CagraSearchParams params) {
    MemorySegment seg = CuVSCagraSearchParams.allocate(resources.getArena());
    CuVSCagraSearchParams.max_queries(seg, params.getMaxQueries());
    CuVSCagraSearchParams.itopk_size(seg, params.getITopKSize());
    CuVSCagraSearchParams.max_iterations(seg, params.getMaxIterations());
    if (params.getCagraSearchAlgo() != null) {
      CuVSCagraSearchParams.algo(seg, params.getCagraSearchAlgo().value);
    }
    CuVSCagraSearchParams.team_size(seg, params.getTeamSize());
    CuVSCagraSearchParams.search_width(seg, params.getSearchWidth());
    CuVSCagraSearchParams.min_iterations(seg, params.getMinIterations());
    CuVSCagraSearchParams.thread_block_size(seg, params.getThreadBlockSize());
    if (params.getHashMapMode() != null) {
      CuVSCagraSearchParams.hashmap_mode(seg, params.getHashMapMode().value);
    }
    CuVSCagraSearchParams.hashmap_max_fill_rate(seg, params.getHashMapMaxFillRate());
    CuVSCagraSearchParams.num_random_samplings(seg, params.getNumRandomSamplings());
    CuVSCagraSearchParams.rand_xor_mask(seg, params.getRandXORMask());
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
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  public static class Builder implements CagraIndex.Builder{

    private float[][] dataset;
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
    public Builder withDataset(float[][] dataset) {
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
        return new CagraIndexImpl(cagraIndexParams, cagraCompressionParams, dataset, cuvsResources);
      }
    }
  }

  /**
   * Holds the memory reference to a CAGRA index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResourcesImpl resources) {
      memorySegment = CuVSCagraIndex.allocate(resources.getArena());
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
