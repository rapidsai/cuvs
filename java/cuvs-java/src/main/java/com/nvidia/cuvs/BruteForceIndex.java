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
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.MethodHandle;
import java.util.UUID;

import com.nvidia.cuvs.common.Util;
import com.nvidia.cuvs.panama.CuVSBruteForceIndex;

import static com.nvidia.cuvs.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.common.Util.checkError;
import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 *
 * {@link BruteForceIndex} encapsulates a BRUTEFORCE index, along with methods
 * to interact with it.
 *
 * @since 25.02
 */
public class BruteForceIndex {

  private static final MethodHandle indexMethodHandle = downcallHandle("build_brute_force_index",
      FunctionDescriptor.of(ADDRESS, ADDRESS, C_LONG, C_LONG, ADDRESS, ADDRESS, C_INT));

  private static final MethodHandle searchMethodHandle = downcallHandle("search_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, C_INT, C_LONG, C_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_LONG, C_LONG));

  private static final MethodHandle destroyIndexMethodHandle = downcallHandle("destroy_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

  private static final MethodHandle serializeMethodHandle = downcallHandle("serialize_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle deserializeMethodHandle = downcallHandle("deserialize_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private final float[][] dataset;
  private final CuVSResources resources;
  private final IndexReference bruteForceIndexReference;
  private final BruteForceIndexParams bruteForceIndexParams;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   *
   * @param dataset               the dataset used for creating the BRUTEFORCE
   *                              index
   * @param resources             an instance of {@link CuVSResources}
   * @param bruteForceIndexParams an instance of {@link BruteForceIndexParams}
   *                              holding the index parameters
   */
  private BruteForceIndex(float[][] dataset, CuVSResources resources, BruteForceIndexParams bruteForceIndexParams)
      throws Throwable {
    this.dataset = dataset;
    this.resources = resources;
    this.bruteForceIndexParams = bruteForceIndexParams;
    this.bruteForceIndexReference = build();
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private BruteForceIndex(InputStream inputStream, CuVSResources resources) throws Throwable {
    this.bruteForceIndexParams = null;
    this.dataset = null;
    this.resources = resources;
    this.bruteForceIndexReference = deserialize(inputStream);
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native destroy_brute_force_index function to de-allocate
   * BRUTEFORCE index
   */
  public void destroyIndex() throws Throwable {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      destroyIndexMethodHandle.invokeExact(bruteForceIndexReference.getMemorySegment(), returnValue);
      checkError(returnValue.get(C_INT, 0L), "destroyIndexMethodHandle");
    }
  }

  /**
   * Invokes the native build_brute_force_index function via the Panama API to
   * build the {@link BruteForceIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = rows > 0 ? dataset[0].length : 0;

    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      MemorySegment dataSeg = Util.buildMemorySegment(localArena, dataset);
      MemorySegment indexSeg = (MemorySegment) indexMethodHandle.invokeExact(
        dataSeg,
        rows,
        cols,
        resources.getMemorySegment(),
        returnValue,
        bruteForceIndexParams.getNumWriterThreads()
      );
      checkError(returnValue.get(C_INT, 0L), "indexMethodHandle");
      return new IndexReference(indexSeg);
    }
  }

  /**
   * Invokes the native search_brute_force_index via the Panama API for searching
   * a BRUTEFORCE index.
   *
   * @param cuvsQuery an instance of {@link BruteForceQuery} holding the query
   *                  vectors and other parameters
   * @return an instance of {@link BruteForceSearchResults} containing the results
   */
  public BruteForceSearchResults search(BruteForceQuery cuvsQuery) throws Throwable {
    checkNotDestroyed();
    long numQueries = cuvsQuery.getQueryVectors().length;
    long numBlocks = cuvsQuery.getTopK() * numQueries;
    int vectorDimension = numQueries > 0 ? cuvsQuery.getQueryVectors()[0].length : 0;
    long prefilterDataLength = cuvsQuery.getPrefilter() != null ? cuvsQuery.getPrefilter().length : 0;
    long numRows = dataset != null ? dataset.length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
    MemorySegment neighborsMemorySegment = resources.getArena().allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.getArena().allocate(distancesSequenceLayout);
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      MemorySegment prefilterDataMemorySegment = cuvsQuery.getPrefilter() != null
              ? Util.buildMemorySegment(localArena, cuvsQuery.getPrefilter())
              : MemorySegment.NULL;
      MemorySegment querySeg = Util.buildMemorySegment(localArena, cuvsQuery.getQueryVectors());
      searchMethodHandle.invokeExact(
        bruteForceIndexReference.getMemorySegment(),
        querySeg,
        cuvsQuery.getTopK(),
        numQueries,
        vectorDimension,
        resources.getMemorySegment(),
        neighborsMemorySegment,
        distancesMemorySegment,
        returnValue,
        prefilterDataMemorySegment,
        prefilterDataLength, numRows
      );
      checkError(returnValue.get(C_INT, 0L), "searchMethodHandle");
    }
    return new BruteForceSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
            distancesMemorySegment, cuvsQuery.getTopK(), cuvsQuery.getMapping(), numQueries);
  }

  /**
   * A method to persist a BRUTEFORCE index using an instance of
   * {@link OutputStream} for writing index bytes.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   */
  public void serialize(OutputStream outputStream) throws Throwable {
    serialize(outputStream, File.createTempFile(UUID.randomUUID().toString(), ".bf"));
  }

  /**
   * A method to persist a BRUTEFORCE index using an instance of
   * {@link OutputStream} and path to the intermediate temporary file.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile     an intermediate {@link File} where BRUTEFORCE index is
   *                     written temporarily
   */
  public void serialize(OutputStream outputStream, File tempFile) throws Throwable {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      MemorySegment pathSeg = Util.buildMemorySegment(localArena, tempFile.getAbsolutePath());
      serializeMethodHandle.invokeExact(
        resources.getMemorySegment(),
        bruteForceIndexReference.getMemorySegment(),
        returnValue,
        pathSeg
      );
      checkError(returnValue.get(C_INT, 0L), "serializeMethodHandle");

      try (FileInputStream fileInputStream = new FileInputStream(tempFile)) {
        fileInputStream.transferTo(outputStream);
      } finally {
        tempFile.delete();
      }
    }
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a BRUTEFORCE
   * index using an {@link InputStream}.
   *
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    checkNotDestroyed();
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".bf";
    IndexReference indexReference = new IndexReference(resources);

    File tempFile = new File(tmpIndexFile);
    try (var in = inputStream;
         FileOutputStream fileOutputStream = new FileOutputStream(tempFile)) {
      in.transferTo(fileOutputStream);
      try (var localArena = Arena.ofConfined()) {
        MemorySegment returnValue = localArena.allocate(C_INT);
        MemorySegment pathSeg = Util.buildMemorySegment(localArena, tmpIndexFile);
        deserializeMethodHandle.invokeExact(
          resources.getMemorySegment(),
          indexReference.getMemorySegment(),
          returnValue,
          pathSeg
        );
        checkError(returnValue.get(C_INT, 0L), "deserializeMethodHandle");
      }
    } finally {
      tempFile.delete();
    }
    return indexReference;
  }

  /**
   * Builder helps configure and create an instance of {@link BruteForceIndex}.
   */
  public static class Builder {

    private float[][] dataset;
    private final CuVSResources cuvsResources;
    private BruteForceIndexParams bruteForceIndexParams;
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
     * Registers an instance of configured {@link BruteForceIndexParams} with this
     * Builder.
     *
     * @param bruteForceIndexParams An instance of BruteForceIndexParams
     * @return An instance of this Builder
     */
    public Builder withIndexParams(BruteForceIndexParams bruteForceIndexParams) {
      this.bruteForceIndexParams = bruteForceIndexParams;
      return this;
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
     * Sets the dataset for building the {@link BruteForceIndex}.
     *
     * @param dataset a two-dimensional float array
     * @return an instance of this Builder
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Builds and returns an instance of {@link BruteForceIndex}.
     *
     * @return an instance of {@link BruteForceIndex}
     */
    public BruteForceIndex build() throws Throwable {
      if (inputStream != null) {
        return new BruteForceIndex(inputStream, cuvsResources);
      } else {
        return new BruteForceIndex(dataset, cuvsResources, bruteForceIndexParams);
      }
    }
  }

  /**
   * Holds the memory reference to a BRUTEFORCE index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResources resources) {
      memorySegment = CuVSBruteForceIndex.allocate(resources.getArena());
    }

    /**
     * Constructs BruteForceIndexReference with an instance of MemorySegment passed
     * as a parameter.
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
