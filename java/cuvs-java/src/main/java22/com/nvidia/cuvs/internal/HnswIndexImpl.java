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

import static com.nvidia.cuvs.internal.CuVSParamsHelper.createHnswIndexParams;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.panama.DLDataType;
import com.nvidia.cuvs.internal.panama.cuvsHnswIndex;
import com.nvidia.cuvs.internal.panama.cuvsHnswIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsHnswSearchParams;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

/**
 * {@link HnswIndex} encapsulates a HNSW index, along with methods to interact
 * with it.
 *
 * @since 25.02
 */
public class HnswIndexImpl implements HnswIndex {

  private final CuVSResources resources;
  private final HnswIndexParams hnswIndexParams;
  private final IndexReference hnswIndexReference;

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private HnswIndexImpl(
      InputStream inputStream, CuVSResources resources, HnswIndexParams hnswIndexParams)
      throws Throwable {
    this.hnswIndexParams = hnswIndexParams;
    this.resources = resources;
    this.hnswIndexReference = deserialize(inputStream);
  }

  /**
   * Invokes the native destroy_hnsw_index to de-allocate the HNSW index
   */
  @Override
  public void close() {
    int returnValue = cuvsHnswIndexDestroy(hnswIndexReference.getMemorySegment());
    checkCuVSError(returnValue, "cuvsHnswIndexDestroy");
  }

  /**
   * Invokes the native search_hnsw_index via the Panama API for searching a HNSW
   * index.
   *
   * @param query an instance of {@link HnswQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link HnswSearchResults} containing the results
   */
  @Override
  public SearchResults search(HnswQuery query) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      int topK = query.getTopK();
      float[][] queryVectors = query.getQueryVectors();
      int numQueries = queryVectors.length;
      long numBlocks = (long) topK * numQueries;
      int vectorDimension = numQueries > 0 ? queryVectors[0].length : 0;

      SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
      SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsMemorySegment = localArena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = localArena.allocate(distancesSequenceLayout);
      MemorySegment querySeg = buildMemorySegment(localArena, queryVectors);

      long[] queriesShape = {numQueries, vectorDimension};
      MemorySegment queriesTensor = prepareTensor(localArena, querySeg, queriesShape, 2, 32, 1, 1);
      long[] neighborsShape = {numQueries, topK};
      MemorySegment neighborsTensor =
          prepareTensor(localArena, neighborsMemorySegment, neighborsShape, 1, 64, 1, 1);
      long[] distancesShape = {numQueries, topK};
      MemorySegment distancesTensor =
          prepareTensor(localArena, distancesMemorySegment, distancesShape, 2, 32, 1, 1);

      try (var resourcesAccessor = query.getResources().access()) {
        var cuvsRes = resourcesAccessor.handle();
        int returnValue = cuvsStreamSync(cuvsRes);
        checkCuVSError(returnValue, "cuvsStreamSync");

        returnValue =
            cuvsHnswSearch(
                cuvsRes,
                segmentFromSearchParams(localArena, query.getHnswSearchParams()),
                hnswIndexReference.getMemorySegment(),
                queriesTensor,
                neighborsTensor,
                distancesTensor);
        checkCuVSError(returnValue, "cuvsHnswSearch");

        returnValue = cuvsStreamSync(cuvsRes);
        checkCuVSError(returnValue, "cuvsStreamSync");
      }

      return HnswSearchResults.create(
          neighborsSequenceLayout,
          distancesSequenceLayout,
          neighborsMemorySegment,
          distancesMemorySegment,
          topK,
          query.getMapping(),
          numQueries);
    }
  }

  private static IndexReference createHnswIndex() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexPtrPtr = localArena.allocate(cuvsHnswIndex_t);
      // cuvsHnswIndexCreate gets a pointer to a cuvsHnswIndex_t, which is defined as a pointer to
      // cuvsHnswIndex.
      // It's basically a "out" parameter: the C functions will create the index and "return back" a
      // pointer to it.
      // The "out parameter" pointer is needed only for the duration of the function invocation (it
      // could be a stack
      // pointer, in C) so we allocate it from our localArena.
      var returnValue = cuvsHnswIndexCreate(indexPtrPtr);
      checkCuVSError(returnValue, "cuvsHnswIndexCreate");
      return new IndexReference(indexPtrPtr.get(cuvsHnswIndex_t, 0));
    }
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a HNSW index
   * using an {@link InputStream}.
   *
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    Path tmpIndexFile =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw")
            .toAbsolutePath();

    try (inputStream;
        var outputStream = Files.newOutputStream(tmpIndexFile);
        var localArena = Arena.ofConfined()) {
      inputStream.transferTo(outputStream);
      MemorySegment pathSeg = buildMemorySegment(localArena, tmpIndexFile.toString());

      var indexReference = createHnswIndex();

      MemorySegment dtype = DLDataType.allocate(localArena);
      DLDataType.bits(dtype, (byte) 32);
      DLDataType.code(dtype, (byte) kDLFloat());
      DLDataType.lanes(dtype, (byte) 1);

      cuvsHnswIndex.dtype(indexReference.memorySegment, dtype);

      try (var params = segmentFromIndexParams(hnswIndexParams);
          var cuvsResourcesAccessor = resources.access()) {
        checkCuVSError(
            cuvsHnswDeserialize(
                cuvsResourcesAccessor.handle(),
                params.handle(),
                pathSeg,
                hnswIndexParams.getVectorDimension(),
                0,
                indexReference.memorySegment),
            "cuvsHnswDeserialize");
      }

      return indexReference;

    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private CloseableHandle segmentFromIndexParams(HnswIndexParams params) {
    var hnswParams = createHnswIndexParams();
    cuvsHnswIndexParams.ef_construction(hnswParams.handle(), params.getEfConstruction());
    cuvsHnswIndexParams.num_threads(hnswParams.handle(), params.getNumThreads());
    return hnswParams;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private static MemorySegment segmentFromSearchParams(Arena arena, HnswSearchParams params) {
    MemorySegment seg = cuvsHnswSearchParams.allocate(arena);
    cuvsHnswSearchParams.ef(seg, params.ef());
    cuvsHnswSearchParams.num_threads(seg, params.numThreads());
    return seg;
  }

  public static HnswIndex.Builder newBuilder(CuVSResources cuvsResources) {
    return new HnswIndexImpl.Builder(Objects.requireNonNull(cuvsResources));
  }

  /**
   * Builder helps configure and create an instance of {@link HnswIndex}.
   */
  public static class Builder implements HnswIndex.Builder {

    private final CuVSResources cuvsResources;
    private InputStream inputStream;
    private HnswIndexParams hnswIndexParams;

    /**
     * Constructs this Builder with an instance of {@link CuVSResources}.
     *
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
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
     * Registers an instance of configured {@link HnswIndexParams} with this
     * Builder.
     *
     * @param hnswIndexParameters An instance of HnswIndexParams.
     * @return An instance of this Builder.
     */
    @Override
    public Builder withIndexParams(HnswIndexParams hnswIndexParameters) {
      this.hnswIndexParams = hnswIndexParameters;
      return this;
    }

    /**
     * Builds and returns an instance of CagraIndex.
     *
     * @return an instance of CagraIndex
     */
    @Override
    public HnswIndexImpl build() throws Throwable {
      return new HnswIndexImpl(inputStream, cuvsResources, hnswIndexParams);
    }
  }

  /**
   * Holds the memory reference to a HNSW index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

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
