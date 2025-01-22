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

/**
 * {@link HnswIndex} encapsulates a HNSW index, along with methods to interact
 * with it.
 * 
 * @since 25.02
 */
public class HnswIndex {

  private final CuVSResources resources;
  private MethodHandle deserializeHnswIndexMethodHandle;
  private MethodHandle searchHnswIndexMethodHandle;
  private MethodHandle destroyHnswIndexMethodHandle;
  private HnswIndexParams hnswIndexParams;
  private IndexReference hnswIndexReference;
  private MemoryLayout longMemoryLayout;
  private MemoryLayout intMemoryLayout;
  private MemoryLayout floatMemoryLayout;

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

    longMemoryLayout = resources.linker.canonicalLayouts().get("long");
    intMemoryLayout = resources.linker.canonicalLayouts().get("int");
    floatMemoryLayout = resources.linker.canonicalLayouts().get("float");

    initializeMethodHandles();
    this.hnswIndexReference = deserialize(inputStream);
  }

  /**
   * Initializes the {@link MethodHandles} for invoking native methods.
   * 
   * @throws IOException @{@link IOException} is unable to load the native library
   */
  private void initializeMethodHandles() throws IOException {
    deserializeHnswIndexMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("deserialize_hnsw_index").get(), FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, intMemoryLayout));

    searchHnswIndexMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("search_hnsw_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, intMemoryLayout, intMemoryLayout,
            longMemoryLayout));

    destroyHnswIndexMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("destroy_hnsw_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
  }

  /**
   * Invokes the native destroy_hnsw_index to de-allocate the HNSW index
   */
  public void destroyIndex() throws Throwable {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    destroyHnswIndexMethodHandle.invokeExact(hnswIndexReference.getMemorySegment(), returnValueMemorySegment);
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

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, longMemoryLayout);
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, floatMemoryLayout);
    MemorySegment neighborsMemorySegment = resources.arena.allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.arena.allocate(distancesSequenceLayout);
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);

    searchHnswIndexMethodHandle.invokeExact(resources.getMemorySegment(), hnswIndexReference.getMemorySegment(),
        query.getHnswSearchParams().getHnswSearchParamsMemorySegment(), returnValueMemorySegment,
        neighborsMemorySegment, distancesMemorySegment,
        Util.buildMemorySegment(resources.linker, resources.arena, query.getQueryVectors()), query.getTopK(),
        vectorDimension, numQueries);

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
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".hnsw";

    File tempFile = new File(tmpIndexFile);
    FileOutputStream fileOutputStream = new FileOutputStream(tempFile);
    byte[] chunk = new byte[bufferLength];
    int chunkLength = 0;
    while ((chunkLength = inputStream.read(chunk)) != -1) {
      fileOutputStream.write(chunk, 0, chunkLength);
    }

    IndexReference indexReference = new IndexReference((MemorySegment) deserializeHnswIndexMethodHandle.invokeExact(
        resources.getMemorySegment(), Util.buildMemorySegment(resources.linker, resources.arena, tmpIndexFile),
        hnswIndexParams.getHnswIndexParamsMemorySegment(), returnValueMemorySegment,
        hnswIndexParams.getVectorDimension()));

    inputStream.close();
    fileOutputStream.close();
    tempFile.delete();

    return indexReference;
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
      memorySegment = CuVSHnswIndex.allocate(resources.arena);
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
