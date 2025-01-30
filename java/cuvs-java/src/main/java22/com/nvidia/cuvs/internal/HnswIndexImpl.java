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

import java.io.FileOutputStream;
import java.io.InputStream;
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

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.CuVSHnswIndex;
import com.nvidia.cuvs.internal.panama.CuVSHnswIndexParams;
import com.nvidia.cuvs.internal.panama.CuVSHnswSearchParams;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.internal.common.Util.checkError;
import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * {@link HnswIndex} encapsulates a HNSW index, along with methods to interact
 * with it.
 *
 * @since 25.02
 */
public class HnswIndexImpl implements HnswIndex {

  private static final MethodHandle deserializeHnswIndexMethodHandle = downcallHandle("deserialize_hnsw_index",
      FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_INT));

  private static final MethodHandle searchHnswIndexMethodHandle = downcallHandle("search_hnsw_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_INT, C_INT, C_LONG));

  private static final MethodHandle destroyHnswIndexMethodHandle = downcallHandle("destroy_hnsw_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

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
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      destroyHnswIndexMethodHandle.invokeExact(hnswIndexReference.getMemorySegment(), returnValue);
      checkError(returnValue.get(C_INT, 0L), "destroyHnswIndexMethodHandle");
    }
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
    long numQueries = query.getQueryVectors().length;
    long numBlocks = query.getTopK() * numQueries;
    int vectorDimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
    MemorySegment neighborsMemorySegment = resources.getArena().allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.getArena().allocate(distancesSequenceLayout);

    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      MemorySegment querySeg = Util.buildMemorySegment(localArena, query.getQueryVectors());
      searchHnswIndexMethodHandle.invokeExact(
        resources.getMemorySegment(),
        hnswIndexReference.getMemorySegment(),
        segmentFromSearchParams(query.getHnswSearchParams()),
        returnValue,
        neighborsMemorySegment,
        distancesMemorySegment,
        querySeg,
        query.getTopK(),
        vectorDimension,
        numQueries
      );
      checkError(returnValue.get(C_INT, 0L), "searchHnswIndexMethodHandle");
    }
    return new HnswSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, query.getTopK(), query.getMapping(), numQueries);
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
    Path tmpIndexFile = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
    tmpIndexFile = tmpIndexFile.toAbsolutePath();

    try (var in = inputStream;
         FileOutputStream fileOutputStream = new FileOutputStream(tmpIndexFile.toFile())) {
      byte[] chunk = new byte[bufferLength];
      int chunkLength;
      while ((chunkLength = in.read(chunk)) != -1) {
        fileOutputStream.write(chunk, 0, chunkLength);
      }

      try (var localArena = Arena.ofConfined()) {
        MemorySegment returnValue = localArena.allocate(C_INT);
        MemorySegment pathSeg = Util.buildMemorySegment(localArena, tmpIndexFile.toString());
        MemorySegment deserSeg = (MemorySegment) deserializeHnswIndexMethodHandle.invokeExact(
          resources.getMemorySegment(),
          pathSeg,
          segmentFromIndexParams(hnswIndexParams),
          returnValue,
          hnswIndexParams.getVectorDimension()
        );
        checkError(returnValue.get(C_INT, 0L), "deserializeHnswIndexMethodHandle");
        return new IndexReference(deserSeg);
      }
    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromIndexParams(HnswIndexParams params) {
    MemorySegment seg = CuVSHnswIndexParams.allocate(resources.getArena());
    CuVSHnswIndexParams.ef_construction(seg, params.getEfConstruction());
    CuVSHnswIndexParams.num_threads(seg, params.getNumThreads());
    return seg;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(HnswSearchParams params) {
    MemorySegment seg = CuVSHnswSearchParams.allocate(resources.getArena());
    CuVSHnswSearchParams.ef(seg, params.getEf());
    CuVSHnswSearchParams.num_threads(seg, params.getNumThreads());
    return seg;
  }

  public static HnswIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }
    return new HnswIndexImpl.Builder((CuVSResourcesImpl)cuvsResources);
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
      memorySegment = CuVSHnswIndex.allocate(resources.getArena());
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
