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

package com.nvidia.cuvs;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.UUID;

import com.nvidia.cuvs.common.Util;
import com.nvidia.cuvs.panama.CuVSHnswIndex;

import static com.nvidia.cuvs.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.common.Util.checkError;
import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * {@link HnswIndex} encapsulates a HNSW index, along with methods to interact
 * with it.
 *
 * @since 25.02
 */
public class HnswIndex {

  private static final MethodHandle deserializeHnswIndexMethodHandle = downcallHandle("deserialize_hnsw_index",
      FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_INT));

  private static final MethodHandle searchHnswIndexMethodHandle = downcallHandle("search_hnsw_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_INT, C_INT, C_LONG));

  private static final MethodHandle destroyHnswIndexMethodHandle = downcallHandle("destroy_hnsw_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

  private final CuVSResources resources;
  private final HnswIndexParams hnswIndexParams;
  private final IndexReference hnswIndexReference;

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private HnswIndex(InputStream inputStream, CuVSResources resources, HnswIndexParams hnswIndexParams)
      throws Throwable {
    this.hnswIndexParams = hnswIndexParams;
    this.resources = resources;
    this.hnswIndexReference = deserialize(inputStream);
  }

  /**
   * Invokes the native destroy_hnsw_index to de-allocate the HNSW index
   */
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
  public HnswSearchResults search(HnswQuery query) throws Throwable {
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
        query.getHnswSearchParams().getHnswSearchParamsMemorySegment(),
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
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".hnsw";

    File tempFile = new File(tmpIndexFile);
    try (var in = inputStream;
         FileOutputStream fileOutputStream = new FileOutputStream(tempFile)) {
      byte[] chunk = new byte[bufferLength];
      int chunkLength;
      while ((chunkLength = in.read(chunk)) != -1) {
        fileOutputStream.write(chunk, 0, chunkLength);
      }

      try (var localArena = Arena.ofConfined()) {
        MemorySegment returnValue = localArena.allocate(C_INT);
        MemorySegment pathSeg = Util.buildMemorySegment(localArena, tmpIndexFile);
        MemorySegment deserSeg = (MemorySegment) deserializeHnswIndexMethodHandle.invokeExact(
          resources.getMemorySegment(),
          pathSeg,
          hnswIndexParams.getHnswIndexParamsMemorySegment(),
          returnValue,
          hnswIndexParams.getVectorDimension()
        );
        checkError(returnValue.get(C_INT, 0L), "deserializeHnswIndexMethodHandle");
        return new IndexReference(deserSeg);
      }
    } finally {
      tempFile.delete();
    }
  }

  /**
   * Builder helps configure and create an instance of {@link HnswIndex}.
   */
  public static class Builder {

    private CuVSResources cuvsResources;
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
    public Builder withIndexParams(HnswIndexParams hnswIndexParameters) {
      this.hnswIndexParams = hnswIndexParameters;
      return this;
    }

    /**
     * Builds and returns an instance of CagraIndex.
     *
     * @return an instance of CagraIndex
     */
    public HnswIndex build() throws Throwable {
      return new HnswIndex(inputStream, cuvsResources, hnswIndexParams);
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
    protected IndexReference(CuVSResources resources) {
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
