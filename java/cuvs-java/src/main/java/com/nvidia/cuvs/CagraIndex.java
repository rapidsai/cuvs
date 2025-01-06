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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.UUID;

import com.nvidia.cuvs.common.Util;
import com.nvidia.cuvs.panama.cuvsCagraIndex;

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
public class CagraIndex {

  private final float[][] dataset;
  private final CuVSResources resources;
  private MethodHandle indexMethodHandle;
  private MethodHandle searchMethodHandle;
  private MethodHandle serializeMethodHandle;
  private MethodHandle deserializeMethodHandle;
  private MethodHandle destroyIndexMethodHandle;
  private MethodHandle serializeCAGRAIndexToHNSWMethodHandle;
  private CagraIndexParams cagraIndexParameters;
  private CagraCompressionParams cagraCompressionParams;
  private IndexReference cagraIndexReference;
  private MemoryLayout longMemoryLayout;
  private MemoryLayout intMemoryLayout;
  private MemoryLayout floatMemoryLayout;

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
  private CagraIndex(CagraIndexParams indexParameters, CagraCompressionParams cagraCompressionParams, float[][] dataset,
      CuVSResources resources) throws Throwable {
    this.cagraIndexParameters = indexParameters;
    this.cagraCompressionParams = cagraCompressionParams;
    this.dataset = dataset;
    this.resources = resources;

    longMemoryLayout = resources.linker.canonicalLayouts().get("long");
    intMemoryLayout = resources.linker.canonicalLayouts().get("int");
    floatMemoryLayout = resources.linker.canonicalLayouts().get("float");

    initializeMethodHandles();
    this.cagraIndexReference = build();
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   * 
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private CagraIndex(InputStream inputStream, CuVSResources resources) throws Throwable {
    this.cagraIndexParameters = null;
    this.cagraCompressionParams = null;
    this.dataset = null;
    this.resources = resources;

    longMemoryLayout = resources.linker.canonicalLayouts().get("long");
    intMemoryLayout = resources.linker.canonicalLayouts().get("int");
    floatMemoryLayout = resources.linker.canonicalLayouts().get("float");

    initializeMethodHandles();
    this.cagraIndexReference = deserialize(inputStream);
  }

  /**
   * Initializes the {@link MethodHandles} for invoking native methods.
   * 
   * @throws IOException @{@link IOException} is unable to load the native library
   */
  private void initializeMethodHandles() throws IOException {
    indexMethodHandle = resources.linker.downcallHandle(resources.getSymbolLookup().find("build_cagra_index").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, longMemoryLayout, longMemoryLayout,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, intMemoryLayout));

    searchMethodHandle = resources.linker.downcallHandle(resources.getSymbolLookup().find("search_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, intMemoryLayout, longMemoryLayout,
            intMemoryLayout, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS));

    serializeMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("serialize_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    deserializeMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("deserialize_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    destroyIndexMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("destroy_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    serializeCAGRAIndexToHNSWMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("serialize_cagra_index_to_hnsw").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

  }

  /**
   * Invokes the native destroy_cagra_index to de-allocate the CAGRA index
   */
  public void destroyIndex() throws Throwable {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    destroyIndexMethodHandle.invokeExact(cagraIndexReference.getMemorySegment(), returnValueMemorySegment);
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

    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);

    MemorySegment indexParamsMemorySegment = cagraIndexParameters != null ? cagraIndexParameters.getMemorySegment()
        : MemorySegment.NULL;

    int numWriterThreads = cagraIndexParameters != null ? cagraIndexParameters.getNumWriterThreads() : 1;

    MemorySegment compressionParamsMemorySegment = cagraCompressionParams != null
        ? cagraCompressionParams.getMemorySegment()
        : MemorySegment.NULL;

    IndexReference indexReference = new IndexReference((MemorySegment) indexMethodHandle.invokeExact(
        Util.buildMemorySegment(resources.linker, resources.arena, dataset), rows, cols, resources.getMemorySegment(),
        returnValueMemorySegment, indexParamsMemorySegment, compressionParamsMemorySegment, numWriterThreads));

    return indexReference;
  }

  /**
   * Invokes the native search_cagra_index via the Panama API for searching a
   * CAGRA index.
   * 
   * @param query an instance of {@link CagraQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link CagraSearchResults} containing the results
   */
  public CagraSearchResults search(CagraQuery query) throws Throwable {
	int topK = query.getMapping() != null ? Math.min(query.getMapping().size(), query.getTopK()): query.getTopK();
    long numQueries = query.getQueryVectors().length;
    long numBlocks = topK * numQueries;
    int vectorDimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, intMemoryLayout);
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, floatMemoryLayout);
    MemorySegment neighborsMemorySegment = resources.arena.allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.arena.allocate(distancesSequenceLayout);
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);

    searchMethodHandle.invokeExact(cagraIndexReference.getMemorySegment(),
        Util.buildMemorySegment(resources.linker, resources.arena, query.getQueryVectors()), topK,
        numQueries, vectorDimension, resources.getMemorySegment(), neighborsMemorySegment, distancesMemorySegment,
        returnValueMemorySegment, query.getCagraSearchParameters().getMemorySegment());

    return new CagraSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, topK, query.getMapping(), numQueries);
  }

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * for writing index bytes.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   */
  public void serialize(OutputStream outputStream) throws Throwable {
    serialize(outputStream, File.createTempFile(UUID.randomUUID().toString(), ".cag"), 1024);
  }

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * for writing index bytes.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  public void serialize(OutputStream outputStream, int bufferLength) throws Throwable {
    serialize(outputStream, File.createTempFile(UUID.randomUUID().toString(), ".cag"), bufferLength);
  }

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * for writing index bytes.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   * @param tempFile     an intermediate {@link File} where CAGRA index is written
   *                     temporarily
   */
  public void serialize(OutputStream outputStream, File tempFile) throws Throwable {
    serialize(outputStream, tempFile, 1024);
  }

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * and path to the intermediate temporary file.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile     an intermediate {@link File} where CAGRA index is written
   *                     temporarily
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  public void serialize(OutputStream outputStream, File tempFile, int bufferLength) throws Throwable {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    serializeMethodHandle.invokeExact(resources.getMemorySegment(), cagraIndexReference.getMemorySegment(),
        returnValueMemorySegment,
        Util.buildMemorySegment(resources.linker, resources.arena, tempFile.getAbsolutePath()));
    FileInputStream fileInputStream = new FileInputStream(tempFile);
    byte[] chunk = new byte[bufferLength];
    int chunkLength = 0;
    while ((chunkLength = fileInputStream.read(chunk)) != -1) {
      outputStream.write(chunk, 0, chunkLength);
    }
    fileInputStream.close();
    tempFile.delete();
  }

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   */
  public void serializeToHNSW(OutputStream outputStream) throws Throwable {
    serializeToHNSW(outputStream, File.createTempFile(UUID.randomUUID().toString(), ".hnsw"), 1024);
  }

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  public void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable {
    serializeToHNSW(outputStream, File.createTempFile(UUID.randomUUID().toString(), ".hnsw"), bufferLength);
  }

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile     an intermediate {@link File} where CAGRA index is written
   *                     temporarily
   */
  public void serializeToHNSW(OutputStream outputStream, File tempFile) throws Throwable {
    serializeToHNSW(outputStream, tempFile, 1024);
  }

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile     an intermediate {@link File} where CAGRA index is written
   *                     temporarily
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  public void serializeToHNSW(OutputStream outputStream, File tempFile, int bufferLength) throws Throwable {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    serializeCAGRAIndexToHNSWMethodHandle.invokeExact(resources.getMemorySegment(),
        Util.buildMemorySegment(resources.linker, resources.arena, tempFile.getAbsolutePath()),
        cagraIndexReference.getMemorySegment(), returnValueMemorySegment);
    FileInputStream fileInputStream = new FileInputStream(tempFile);
    byte[] chunk = new byte[bufferLength];
    int chunkLength = 0;
    while ((chunkLength = fileInputStream.read(chunk)) != -1) {
      outputStream.write(chunk, 0, chunkLength);
    }
    fileInputStream.close();
    tempFile.delete();
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
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".cag";
    IndexReference indexReference = new IndexReference(resources);

    File tempFile = new File(tmpIndexFile);
    FileOutputStream fileOutputStream = new FileOutputStream(tempFile);
    byte[] chunk = new byte[bufferLength];
    int chunkLength = 0;
    while ((chunkLength = inputStream.read(chunk)) != -1) {
      fileOutputStream.write(chunk, 0, chunkLength);
    }
    deserializeMethodHandle.invokeExact(resources.getMemorySegment(), indexReference.getMemorySegment(),
        returnValueMemorySegment, Util.buildMemorySegment(resources.linker, resources.arena, tmpIndexFile));

    inputStream.close();
    fileOutputStream.close();
    tempFile.delete();

    return indexReference;
  }

  /**
   * Gets an instance of {@link CagraIndexParams}
   * 
   * @return an instance of {@link CagraIndexParams}
   */
  public CagraIndexParams getCagraIndexParameters() {
    return cagraIndexParameters;
  }

  /**
   * Gets an instance of {@link CuVSResources}
   * 
   * @return an instance of {@link CuVSResources}
   */
  public CuVSResources getCuVSResources() {
    return resources;
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  public static class Builder {

    private float[][] dataset;
    private CagraIndexParams cagraIndexParams;
    private CagraCompressionParams cagraCompressionParams;
    private CuVSResources cuvsResources;
    private InputStream inputStream;

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
     * Sets the dataset for building the {@link CagraIndex}.
     * 
     * @param dataset a two-dimensional float array
     * @return an instance of this Builder
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Registers an instance of configured {@link CagraIndexParams} with this
     * Builder.
     * 
     * @param cagraIndexParameters An instance of CagraIndexParams.
     * @return An instance of this Builder.
     */
    public Builder withIndexParams(CagraIndexParams cagraIndexParameters) {
      this.cagraIndexParams = cagraIndexParameters;
      return this;
    }

    /**
     * Registers an instance of configured {@link CagraCompressionParams} with this
     * Builder.
     * 
     * @param cagraCompressionParams An instance of CagraCompressionParams.
     * @return An instance of this Builder.
     */
    public Builder withCompressionParams(CagraCompressionParams cagraCompressionParams) {
      this.cagraCompressionParams = cagraCompressionParams;
      return this;
    }

    /**
     * Builds and returns an instance of CagraIndex.
     * 
     * @return an instance of CagraIndex
     */
    public CagraIndex build() throws Throwable {
      if (inputStream != null) {
        return new CagraIndex(inputStream, cuvsResources);
      } else {
        return new CagraIndex(cagraIndexParams, cagraCompressionParams, dataset, cuvsResources);
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
    protected IndexReference(CuVSResources resources) {
      memorySegment = cuvsCagraIndex.allocate(resources.arena);
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
