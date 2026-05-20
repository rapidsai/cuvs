/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.DEVICE_TO_HOST;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.HOST_TO_DEVICE;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpyAsync;
import static com.nvidia.cuvs.internal.common.Util.getStream;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraSearchMultiPartition;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLFloat;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLInt;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLUInt;

import com.nvidia.cuvs.FilterBitsetHandle.DeviceData;
import com.nvidia.cuvs.internal.BufferedCagraSearch;
import com.nvidia.cuvs.internal.CuVSMatrixInternal;
import com.nvidia.cuvs.internal.CuVSParamsHelper;
import com.nvidia.cuvs.internal.SelectKHelper;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

/**
 * Performs a single-query approximate nearest neighbor search across multiple CAGRA index
 * partitions using a shared GPU buffer, eliminating per-partition device-to-host copies.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Allocate two global device buffers sized {@code numPartitions × k}:
 *       one for uint32 neighbor ordinals and one for float32 distances.</li>
 *   <li>Call {@code cuvsCagraSearchMultiPartition} which launches a single GPU kernel covering all
 *       partitions concurrently (one CTA per partition), writing results into the global buffers.</li>
 *   <li>Call {@code cuvsSelectK} on the main stream to find the global top-k entirely on GPU.</li>
 *   <li>Sync the main stream.</li>
 *   <li>Copy the three result arrays to host in a single pass.</li>
 *   <li>Decode each result: {@code partition = position / k},
 *       {@code ordinal = ordinals[position]}.</li>
 * </ol>
 *
 * @since 25.10
 */
public class MultiPartitionCagraSearch {

  private MultiPartitionCagraSearch() {}

  /**
   * Searches multiple CAGRA index partitions for the global top-k nearest neighbors.
   *
   * <p>Per-partition prefilters (if any) are read from {@link CagraQuery#getPrefilter()}. For
   * repeated queries with the same filter, prefer the overload that accepts a
   * {@link FilterBitsetHandle} to avoid redundant host-side bitset construction and H2D transfers.
   *
   * @param resources shared {@link CuVSResources} handle
   * @param indices   one {@link CagraIndex} per partition, in partition order
   * @param queries   one {@link CagraQuery} per partition; may carry per-partition prefilter BitSets
   * @param k         number of global nearest neighbors to return
   */
  public static MultiPartitionSearchResults search(
      CuVSResources resources, List<CagraIndex> indices, List<CagraQuery> queries, int k)
      throws Throwable {
    return search(resources, indices, queries, k, /* filter= */ null);
  }

  /**
   * Searches multiple CAGRA index partitions with a pre-cached device-side filter.
   *
   * <p>When {@code filter} is non-null, prefilters on the {@code queries} are ignored; the filter
   * is applied via the pre-uploaded combined bitset in {@code filter}. This overload avoids both
   * the host-side O(N) bit evaluation and the H2D transfer on cache hits.
   *
   * @param resources shared {@link CuVSResources} handle
   * @param indices   one {@link CagraIndex} per partition, in partition order
   * @param queries   one {@link CagraQuery} per partition (prefilters ignored when filter != null)
   * @param k         number of global nearest neighbors to return
   * @param filter    pre-built combined bitset handle, or {@code null} for unfiltered search
   */
  public static MultiPartitionSearchResults search(
      CuVSResources resources,
      List<CagraIndex> indices,
      List<CagraQuery> queries,
      int k,
      FilterBitsetHandle filter)
      throws Throwable {
    int numPartitions = indices.size();
    if (numPartitions != queries.size()) {
      throw new IllegalArgumentException(
          "indices and queries must have the same size; got "
              + numPartitions
              + " vs "
              + queries.size());
    }
    if (numPartitions == 0) {
      return new MultiPartitionSearchResults(0, new int[0], new int[0], new float[0]);
    }

    BufferedCagraSearch[] buffered = new BufferedCagraSearch[numPartitions];
    for (int i = 0; i < numPartitions; i++) {
      CagraIndex idx = indices.get(i);
      if (!(idx instanceof BufferedCagraSearch)) {
        throw new IllegalArgumentException(
            "Index at position " + i + " does not support buffered search");
      }
      buffered[i] = (BufferedCagraSearch) idx;
    }

    long totalCandidates = (long) numPartitions * k;
    long neighborsBytes = totalCandidates * Integer.BYTES;
    long distancesBytes = totalCandidates * Float.BYTES;
    long outIdxBytes = (long) k * Long.BYTES;
    long outValBytes = (long) k * Float.BYTES;

    CagraSearchParams searchParameters = queries.get(0).getCagraSearchParameters();

    // When no pre-built handle is supplied, fall back to reading BitSets from queries.
    boolean useQueryBitsets = (filter == null);
    boolean hasQueryFilter = false;
    long[] partBitOffsets = null;
    long totalBits = 0;
    long[] combinedLongs = null;

    if (useQueryBitsets) {
      for (int i = 0; i < numPartitions; i++) {
        if (queries.get(i).getPrefilter() != null) {
          hasQueryFilter = true;
          break;
        }
      }
      if (hasQueryFilter) {
        partBitOffsets = new long[numPartitions];
        for (int i = 0; i < numPartitions; i++) {
          partBitOffsets[i] = totalBits;
          int nd = queries.get(i).getNumDocs();
          totalBits += ((long) (nd + 63) / 64) * 64;
        }
        combinedLongs = new long[(int) (totalBits / 64)];
        for (int i = 0; i < numPartitions; i++) {
          BitSet bs = queries.get(i).getPrefilter();
          int nd = queries.get(i).getNumDocs();
          int longOffset = (int) (partBitOffsets[i] / 64);
          packBitset(bs, nd, combinedLongs, longOffset);
        }
      }
    }

    long combinedBitsetBytes =
        (useQueryBitsets && hasQueryFilter) ? (long) combinedLongs.length * Long.BYTES : 0;
    long partOffsetsBytes =
        (useQueryBitsets && hasQueryFilter) ? (long) numPartitions * Long.BYTES : 0;

    try (var resourcesAccessor = resources.access()) {
      long cuvsRes = resourcesAccessor.handle();
      var cuvsStream = getStream(cuvsRes);

      // Per-call device allocations for neighbors, distances, and selectK outputs.
      // When using a FilterBitsetHandle, bitset device memory is owned by the handle (not freed
      // here), so combinedBitsetDP / partOffsetsDP are null in that path.
      try (var globalNeighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
          var globalDistancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
          var outIdxDP = allocateRMMSegment(cuvsRes, outIdxBytes);
          var outValDP = allocateRMMSegment(cuvsRes, outValBytes);
          var combinedBitsetDP =
              (useQueryBitsets && hasQueryFilter)
                  ? allocateRMMSegment(cuvsRes, combinedBitsetBytes)
                  : null;
          var partOffsetsDP =
              (useQueryBitsets && hasQueryFilter)
                  ? allocateRMMSegment(cuvsRes, partOffsetsBytes)
                  : null) {

        // filterHostArena is non-null only on the slow path (cache miss). It must outlive the
        // stream-ordered H2D copies, so it is closed in the finally below — after cuvsStreamSync
        // guarantees all prior stream operations (including the H2D copies) have completed.
        Arena filterHostArena = null;
        try {
          try (var arena = Arena.ofConfined()) {
            MemorySegment sp = CuVSParamsHelper.buildCagraSearchParams(arena, searchParameters);

            MemorySegment indexArray = arena.allocate(ValueLayout.ADDRESS, numPartitions);
            MemorySegment queriesArray = arena.allocate(ValueLayout.ADDRESS, numPartitions);
            MemorySegment neighborsArray = arena.allocate(ValueLayout.ADDRESS, numPartitions);
            MemorySegment distancesArray = arena.allocate(ValueLayout.ADDRESS, numPartitions);

            long[] partShape = {1, k};
            for (int i = 0; i < numPartitions; i++) {
              indexArray.setAtIndex(ValueLayout.ADDRESS, i, buffered[i].getIndexHandle());

              var queryVectors = (CuVSMatrixInternal) queries.get(i).getQueryVectors();
              queriesArray.setAtIndex(ValueLayout.ADDRESS, i, queryVectors.toTensor(arena));

              long nByteOffset = (long) i * k * Integer.BYTES;
              MemorySegment nSlice =
                  MemorySegment.ofAddress(globalNeighborsDP.handle().address() + nByteOffset);
              neighborsArray.setAtIndex(
                  ValueLayout.ADDRESS,
                  i,
                  prepareTensor(arena, nSlice, partShape, kDLUInt(), 32, kDLCUDA()));

              long dByteOffset = (long) i * k * Float.BYTES;
              MemorySegment dSlice =
                  MemorySegment.ofAddress(globalDistancesDP.handle().address() + dByteOffset);
              distancesArray.setAtIndex(
                  ValueLayout.ADDRESS,
                  i,
                  prepareTensor(arena, dSlice, partShape, kDLFloat(), 32, kDLCUDA()));
            }

            // Build cuvsFilter: either from pre-uploaded handle (cache hit → no H2D) or
            // from per-call RMM allocations with a stream-ordered async H2D transfer.
            MemorySegment filterSeg = cuvsFilter.allocate(arena);
            if (filter != null) {
              // Fast path: device data already uploaded for this thread.
              DeviceData dev = filter.getOrUpload(cuvsRes);
              buildCuvsFilterStruct(
                  arena,
                  filterSeg,
                  dev.combinedBitsetDP.handle(),
                  dev.partOffsetsDP.handle(),
                  dev.totalBits,
                  dev.numPartitions);
            } else if (hasQueryFilter) {
              // Slow path: upload from query BitSets (first call or cache miss).
              // filterHostArena is kept alive until after cuvsStreamSync (see outer finally).
              filterHostArena = Arena.ofConfined();
              MemorySegment hostBitset = filterHostArena.allocate(combinedBitsetBytes, Long.BYTES);
              MemorySegment.copy(
                  combinedLongs, 0, hostBitset, ValueLayout.JAVA_LONG, 0, combinedLongs.length);
              cudaMemcpyAsync(
                  combinedBitsetDP.handle(),
                  hostBitset,
                  combinedBitsetBytes,
                  HOST_TO_DEVICE,
                  cuvsStream);

              MemorySegment hostOffsets = filterHostArena.allocate(partOffsetsBytes, Long.BYTES);
              MemorySegment.copy(
                  partBitOffsets, 0, hostOffsets, ValueLayout.JAVA_LONG, 0, numPartitions);
              cudaMemcpyAsync(
                  partOffsetsDP.handle(),
                  hostOffsets,
                  partOffsetsBytes,
                  HOST_TO_DEVICE,
                  cuvsStream);

              buildCuvsFilterStruct(
                  arena,
                  filterSeg,
                  combinedBitsetDP.handle(),
                  partOffsetsDP.handle(),
                  totalBits,
                  numPartitions);
            } else {
              cuvsFilter.type(filterSeg, 0 /* NO_FILTER */);
              cuvsFilter.addr(filterSeg, 0L);
            }

            checkCuVSError(
                cuvsCagraSearchMultiPartition(
                    cuvsRes,
                    sp,
                    numPartitions,
                    indexArray,
                    queriesArray,
                    neighborsArray,
                    distancesArray,
                    filterSeg),
                "cuvsCagraSearchMultiPartition");
          }

          // Select global top-k on GPU.
          SelectKHelper.selectK(
              cuvsRes,
              globalDistancesDP.handle(),
              totalCandidates,
              outValDP.handle(),
              outIdxDP.handle(),
              k);

          try (var hostArena = Arena.ofConfined()) {
            MemorySegment hostBuf =
                hostArena.allocate(outIdxBytes + outValBytes + neighborsBytes, Long.BYTES);
            MemorySegment hostOutIdx = hostBuf.asSlice(0, outIdxBytes);
            MemorySegment hostOutVal = hostBuf.asSlice(outIdxBytes, outValBytes);
            MemorySegment hostAllOrdinals =
                hostBuf.asSlice(outIdxBytes + outValBytes, neighborsBytes);

            cudaMemcpyAsync(hostOutIdx, outIdxDP.handle(), outIdxBytes, DEVICE_TO_HOST, cuvsStream);
            cudaMemcpyAsync(hostOutVal, outValDP.handle(), outValBytes, DEVICE_TO_HOST, cuvsStream);
            cudaMemcpyAsync(
                hostAllOrdinals,
                globalNeighborsDP.handle(),
                neighborsBytes,
                DEVICE_TO_HOST,
                cuvsStream);

            checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync after D2H copy");

            int[] partitionIndices = new int[k];
            int[] selectedOrdinals = new int[k];
            float[] selectedDistances = new float[k];
            int count = 0;

            for (int j = 0; j < k; j++) {
              long pos = hostOutIdx.getAtIndex(ValueLayout.JAVA_LONG, j);
              float dist = hostOutVal.getAtIndex(ValueLayout.JAVA_FLOAT, j);
              int ordinal = hostAllOrdinals.getAtIndex(ValueLayout.JAVA_INT, (int) pos);

              if (ordinal < 0) continue;
              partitionIndices[count] = (int) (pos / k);
              selectedOrdinals[count] = ordinal;
              selectedDistances[count] = dist;
              count++;
            }

            return new MultiPartitionSearchResults(
                count, partitionIndices, selectedOrdinals, selectedDistances);
          }
        } finally {
          if (filterHostArena != null) filterHostArena.close();
        }
      }
    }
  }

  /**
   * Populates a {@code cuvsFilter} MemorySegment for a MULTI_PARTITION_BITSET filter using
   * pre-uploaded device buffers.
   */
  private static void buildCuvsFilterStruct(
      Arena arena,
      MemorySegment filterSeg,
      MemorySegment combinedBitsetHandle,
      MemorySegment partOffsetsHandle,
      long totalBits,
      int numPartitions) {
    long[] bitsetShape = {(totalBits + 31) / 32};
    MemorySegment combinedBitsetTensor =
        prepareTensor(arena, combinedBitsetHandle, bitsetShape, kDLUInt(), 32, kDLCUDA());
    long[] offsetsShape = {numPartitions};
    MemorySegment partOffsetsTensor =
        prepareTensor(arena, partOffsetsHandle, offsetsShape, kDLInt(), 64, kDLCUDA());

    // cuvsMultiPartitionBitsetFilter:
    //   {ptr combined_bitset, int64 total_bits, ptr partition_offsets}
    MemorySegment mpbFilter = arena.allocate(24, 8);
    mpbFilter.set(ValueLayout.JAVA_LONG, 0, combinedBitsetTensor.address());
    mpbFilter.set(ValueLayout.JAVA_LONG, 8, totalBits);
    mpbFilter.set(ValueLayout.JAVA_LONG, 16, partOffsetsTensor.address());

    cuvsFilter.type(filterSeg, 3 /* MULTI_PARTITION_BITSET */);
    cuvsFilter.addr(filterSeg, mpbFilter.address());
  }

  /**
   * Packs {@code numDocs} bits from {@code bs} (or all-ones if {@code bs} is null) into
   * {@code dest} starting at long index {@code destLongOffset}.
   */
  private static void packBitset(BitSet bs, int numDocs, long[] dest, int destLongOffset) {
    int numLongs = (numDocs + 63) / 64;
    if (bs == null) {
      Arrays.fill(dest, destLongOffset, destLongOffset + numLongs, -1L);
      int tail = numDocs % 64;
      if (tail != 0) {
        dest[destLongOffset + numLongs - 1] = (1L << tail) - 1L;
      }
    } else {
      long[] bsLongs = bs.toLongArray();
      int copyLen = Math.min(bsLongs.length, numLongs);
      System.arraycopy(bsLongs, 0, dest, destLongOffset, copyLen);
    }
  }
}
