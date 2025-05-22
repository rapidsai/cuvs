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
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsHnswDeserialize;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsHnswIndexCreate;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsHnswIndexDestroy;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsHnswSearch;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsResources_t;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI.cuvsStreamSync;

import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.panama.DLDataType;
import com.nvidia.cuvs.internal.panama.cuvsHnswIndex;
import com.nvidia.cuvs.internal.panama.cuvsHnswIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsHnswSearchParams;

/**
 * {@link HnswIndex} encapsulates a HNSW index, along with methods to interact
 * with it.
 *
 * @since 25.02
 */
public class HnswIndexImpl implements HnswIndex {

  private final CuVSResourcesImpl resources;
  private final HnswIndexParams hnswIndexParams;
  private final IndexReference hnswIndexReference;

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResourcesImpl}
   */
  private HnswIndexImpl(InputStream inputStream, CuVSResourcesImpl resources, HnswIndexParams hnswIndexParams)
      throws Throwable {
    this.hnswIndexParams = hnswIndexParams;
    this.resources = resources;
    this.hnswIndexReference = deserialize(inputStream);
  }

  /**
   * Invokes the native destroy_hnsw_index to de-allocate the HNSW index
   */
  @Override
  public void destroyIndex() throws Throwable {
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
      long numBlocks = topK * numQueries;
      int vectorDimension = numQueries > 0 ? queryVectors[0].length : 0;
      Arena arena = resources.getArena();

      SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
      SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsMemorySegment = arena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = arena.allocate(distancesSequenceLayout);
      MemorySegment querySeg = buildMemorySegment(arena, queryVectors);

      long cuvsRes = resources.getMemorySegment().get(cuvsResources_t, 0);

      long queriesShape[] = { numQueries, vectorDimension };
      MemorySegment queriesTensor = prepareTensor(arena, querySeg, queriesShape, 2, 32, 2, 1, 1);
      long neighborsShape[] = { numQueries, topK };
      MemorySegment neighborsTensor = prepareTensor(arena, neighborsMemorySegment, neighborsShape, 1, 64, 2, 1, 1);
      long distancesShape[] = { numQueries, topK };
      MemorySegment distancesTensor = prepareTensor(arena, distancesMemorySegment, distancesShape, 2, 32, 2, 1, 1);

      int returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      returnValue = cuvsHnswSearch(cuvsRes, segmentFromSearchParams(query.getHnswSearchParams()),
          hnswIndexReference.getMemorySegment(), queriesTensor, neighborsTensor, distancesTensor);
      checkCuVSError(returnValue, "cuvsHnswSearch");

      returnValue = cuvsStreamSync(cuvsRes);
      checkCuVSError(returnValue, "cuvsStreamSync");

      return new HnswSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
          distancesMemorySegment, topK, query.getMapping(), numQueries);
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
    return deserialize(inputStream, 1024);
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a HNSW index
   * using an {@link InputStream}.
   *
   * @param inputStream  an instance of {@link InputStream}
   * @param bufferLength the length of the buffer to use while reading the bytes
   *                     from the stream. Default value is 1024.
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream, int bufferLength) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      Path tmpIndexFile = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
      tmpIndexFile = tmpIndexFile.toAbsolutePath();

      try (var in = inputStream; FileOutputStream fileOutputStream = new FileOutputStream(tmpIndexFile.toFile())) {
        byte[] chunk = new byte[bufferLength];
        int chunkLength;
        while ((chunkLength = in.read(chunk)) != -1) {
          fileOutputStream.write(chunk, 0, chunkLength);
        }

        Arena arena = resources.getArena();
        MemorySegment pathSeg = buildMemorySegment(arena, tmpIndexFile.toString());

        long cuvsRes = resources.getMemorySegment().get(cuvsResources_t, 0);
        MemorySegment hnswIndex = cuvsHnswIndex.allocate(arena);
        int returnValue = cuvsHnswIndexCreate(hnswIndex);
        checkCuVSError(returnValue, "cuvsHnswIndexCreate");

        MemorySegment dtype = DLDataType.allocate(arena);
        DLDataType.bits(dtype, (byte) 32);
        DLDataType.code(dtype, (byte) 2); // kDLFloat
        DLDataType.lanes(dtype, (byte) 1);

        cuvsHnswIndex.dtype(hnswIndex, dtype);

        returnValue = cuvsHnswDeserialize(cuvsRes, segmentFromIndexParams(hnswIndexParams), pathSeg,
            hnswIndexParams.getVectorDimension(), 0, hnswIndex);
        checkCuVSError(returnValue, "cuvsHnswDeserialize");

        return new IndexReference(hnswIndex);

      } finally {
        Files.deleteIfExists(tmpIndexFile);
      }
    }
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromIndexParams(HnswIndexParams params) {
    MemorySegment seg = cuvsHnswIndexParams.allocate(resources.getArena());
    cuvsHnswIndexParams.ef_construction(seg, params.getEfConstruction());
    cuvsHnswIndexParams.num_threads(seg, params.getNumThreads());
    return seg;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(HnswSearchParams params) {
    MemorySegment seg = cuvsHnswSearchParams.allocate(resources.getArena());
    cuvsHnswSearchParams.ef(seg, params.ef());
    cuvsHnswSearchParams.num_threads(seg, params.numThreads());
    return seg;
  }

  public static HnswIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }
    return new HnswIndexImpl.Builder((CuVSResourcesImpl) cuvsResources);
  }

  /**
   * Builder helps configure and create an instance of {@link HnswIndex}.
   */
  public static class Builder implements HnswIndex.Builder {

    private final CuVSResourcesImpl cuvsResources;
    private InputStream inputStream;
    private HnswIndexParams hnswIndexParams;

    /**
     * Constructs this Builder with an instance of {@link CuVSResources}.
     *
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResourcesImpl cuvsResources) {
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
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResourcesImpl resources) {
      memorySegment = cuvsHnswIndex.allocate(resources.getArena());
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
