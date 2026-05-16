/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraSearchMultiSegment;
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
 * Performs a single-query approximate nearest neighbor search across multiple CAGRA index segments
 * using a shared GPU buffer, eliminating per-segment device-to-host copies.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Allocate two global device buffers sized {@code numSegments × k}:
 *       one for uint32 neighbor ordinals and one for float32 distances.</li>
 *   <li>Call {@code cuvsCagraSearchMultiSegment} which launches a single GPU kernel covering all
 *       segments concurrently (one CTA per segment), writing results into the global buffers.</li>
 *   <li>Call {@code cuvsSelectK} on the main stream to find the global top-k entirely on GPU.</li>
 *   <li>Sync the main stream.</li>
 *   <li>Copy the three result arrays to host in a single pass.</li>
 *   <li>Decode each result: {@code segment = position / k},
 *       {@code ordinal = ordinals[position]}.</li>
 * </ol>
 *
 * @since 25.10
 */
public class MultiSegmentCagraSearch {

  private MultiSegmentCagraSearch() {}

  /**
   * Searches multiple CAGRA index segments for the global top-k nearest neighbors.
   *
   * <p>Per-segment prefilters (if any) are read from {@link CagraQuery#getPrefilter()}. For
   * repeated queries with the same filter, prefer the overload that accepts a
   * {@link FilterBitsetHandle} to avoid redundant host-side bitset construction and H2D transfers.
   *
   * @param resources shared {@link CuVSResources} handle
   * @param indices   one {@link CagraIndex} per segment, in segment order
   * @param queries   one {@link CagraQuery} per segment; may carry per-segment prefilter BitSets
   * @param k         number of global nearest neighbors to return
   */
  public static MultiSegmentSearchResults search(
      CuVSResources resources, List<CagraIndex> indices, List<CagraQuery> queries, int k)
      throws Throwable {
    return search(resources, indices, queries, k, /* filter= */ null);
  }

  /**
   * Searches multiple CAGRA index segments with a pre-cached device-side filter.
   *
   * <p>When {@code filter} is non-null, prefilters on the {@code queries} are ignored; the filter
   * is applied via the pre-uploaded combined bitset in {@code filter}. This overload avoids both
   * the host-side O(N) bit evaluation and the H2D transfer on cache hits.
   *
   * @param resources shared {@link CuVSResources} handle
   * @param indices   one {@link CagraIndex} per segment, in segment order
   * @param queries   one {@link CagraQuery} per segment (prefilters ignored when filter != null)
   * @param k         number of global nearest neighbors to return
   * @param filter    pre-built combined bitset handle, or {@code null} for unfiltered search
   */
  public static MultiSegmentSearchResults search(
      CuVSResources resources,
      List<CagraIndex> indices,
      List<CagraQuery> queries,
      int k,
      FilterBitsetHandle filter)
      throws Throwable {
    int numSegments = indices.size();
    if (numSegments != queries.size()) {
      throw new IllegalArgumentException(
          "indices and queries must have the same size; got "
              + numSegments
              + " vs "
              + queries.size());
    }
    if (numSegments == 0) {
      return new MultiSegmentSearchResults(0, new int[0], new int[0], new float[0]);
    }

    BufferedCagraSearch[] buffered = new BufferedCagraSearch[numSegments];
    for (int i = 0; i < numSegments; i++) {
      CagraIndex idx = indices.get(i);
      if (!(idx instanceof BufferedCagraSearch)) {
        throw new IllegalArgumentException(
            "Index at position " + i + " does not support buffered search");
      }
      buffered[i] = (BufferedCagraSearch) idx;
    }

    long totalCandidates = (long) numSegments * k;
    long neighborsBytes = totalCandidates * Integer.BYTES;
    long distancesBytes = totalCandidates * Float.BYTES;
    long outIdxBytes = (long) k * Long.BYTES;
    long outValBytes = (long) k * Float.BYTES;

    CagraSearchParams searchParameters = queries.get(0).getCagraSearchParameters();

    // When no pre-built handle is supplied, fall back to reading BitSets from queries.
    boolean useQueryBitsets = (filter == null);
    boolean hasQueryFilter = false;
    long[] segBitOffsets = null;
    long totalBits = 0;
    long[] combinedLongs = null;

    if (useQueryBitsets) {
      for (int i = 0; i < numSegments; i++) {
        if (queries.get(i).getPrefilter() != null) {
          hasQueryFilter = true;
          break;
        }
      }
      if (hasQueryFilter) {
        segBitOffsets = new long[numSegments];
        for (int i = 0; i < numSegments; i++) {
          segBitOffsets[i] = totalBits;
          int nd = queries.get(i).getNumDocs();
          totalBits += ((long) (nd + 63) / 64) * 64;
        }
        combinedLongs = new long[(int) (totalBits / 64)];
        for (int i = 0; i < numSegments; i++) {
          BitSet bs = queries.get(i).getPrefilter();
          int nd = queries.get(i).getNumDocs();
          int longOffset = (int) (segBitOffsets[i] / 64);
          packBitset(bs, nd, combinedLongs, longOffset);
        }
      }
    }

    long combinedBitsetBytes =
        (useQueryBitsets && hasQueryFilter) ? (long) combinedLongs.length * Long.BYTES : 0;
    long segOffsetsBytes =
        (useQueryBitsets && hasQueryFilter) ? (long) numSegments * Long.BYTES : 0;

    try (var resourcesAccessor = resources.access()) {
      long cuvsRes = resourcesAccessor.handle();
      var cuvsStream = getStream(cuvsRes);

      // Per-call device allocations for neighbors, distances, and selectK outputs.
      // When using a FilterBitsetHandle, bitset device memory is owned by the handle (not freed
      // here), so combinedBitsetDP / segOffsetsDP are null in that path.
      try (var globalNeighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
          var globalDistancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
          var outIdxDP = allocateRMMSegment(cuvsRes, outIdxBytes);
          var outValDP = allocateRMMSegment(cuvsRes, outValBytes);
          var combinedBitsetDP =
              (useQueryBitsets && hasQueryFilter)
                  ? allocateRMMSegment(cuvsRes, combinedBitsetBytes)
                  : null;
          var segOffsetsDP =
              (useQueryBitsets && hasQueryFilter)
                  ? allocateRMMSegment(cuvsRes, segOffsetsBytes)
                  : null) {

        // filterHostArena is non-null only on the slow path (cache miss). It must outlive the
        // stream-ordered H2D copies, so it is closed in the finally below — after cuvsStreamSync
        // guarantees all prior stream operations (including the H2D copies) have completed.
        Arena filterHostArena = null;
        try {
          try (var arena = Arena.ofConfined()) {
            MemorySegment sp = CuVSParamsHelper.buildCagraSearchParams(arena, searchParameters);

            MemorySegment indexArray = arena.allocate(ValueLayout.ADDRESS, numSegments);
            MemorySegment queriesArray = arena.allocate(ValueLayout.ADDRESS, numSegments);
            MemorySegment neighborsArray = arena.allocate(ValueLayout.ADDRESS, numSegments);
            MemorySegment distancesArray = arena.allocate(ValueLayout.ADDRESS, numSegments);

            long[] segShape = {1, k};
            for (int i = 0; i < numSegments; i++) {
              indexArray.setAtIndex(ValueLayout.ADDRESS, i, buffered[i].getIndexHandle());

              var queryVectors = (CuVSMatrixInternal) queries.get(i).getQueryVectors();
              queriesArray.setAtIndex(ValueLayout.ADDRESS, i, queryVectors.toTensor(arena));

              long nByteOffset = (long) i * k * Integer.BYTES;
              MemorySegment nSlice =
                  MemorySegment.ofAddress(globalNeighborsDP.handle().address() + nByteOffset);
              neighborsArray.setAtIndex(
                  ValueLayout.ADDRESS,
                  i,
                  prepareTensor(arena, nSlice, segShape, kDLUInt(), 32, kDLCUDA()));

              long dByteOffset = (long) i * k * Float.BYTES;
              MemorySegment dSlice =
                  MemorySegment.ofAddress(globalDistancesDP.handle().address() + dByteOffset);
              distancesArray.setAtIndex(
                  ValueLayout.ADDRESS,
                  i,
                  prepareTensor(arena, dSlice, segShape, kDLFloat(), 32, kDLCUDA()));
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
                  dev.segOffsetsDP.handle(),
                  dev.totalBits,
                  dev.numSegments);
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

              MemorySegment hostOffsets = filterHostArena.allocate(segOffsetsBytes, Long.BYTES);
              MemorySegment.copy(
                  segBitOffsets, 0, hostOffsets, ValueLayout.JAVA_LONG, 0, numSegments);
              cudaMemcpyAsync(
                  segOffsetsDP.handle(), hostOffsets, segOffsetsBytes, HOST_TO_DEVICE, cuvsStream);

              buildCuvsFilterStruct(
                  arena,
                  filterSeg,
                  combinedBitsetDP.handle(),
                  segOffsetsDP.handle(),
                  totalBits,
                  numSegments);
            } else {
              cuvsFilter.type(filterSeg, 0 /* NO_FILTER */);
              cuvsFilter.addr(filterSeg, 0L);
            }

            checkCuVSError(
                cuvsCagraSearchMultiSegment(
                    cuvsRes,
                    sp,
                    numSegments,
                    indexArray,
                    queriesArray,
                    neighborsArray,
                    distancesArray,
                    filterSeg),
                "cuvsCagraSearchMultiSegment");
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

            int[] segmentIndices = new int[k];
            int[] selectedOrdinals = new int[k];
            float[] selectedDistances = new float[k];
            int count = 0;

            for (int j = 0; j < k; j++) {
              long pos = hostOutIdx.getAtIndex(ValueLayout.JAVA_LONG, j);
              float dist = hostOutVal.getAtIndex(ValueLayout.JAVA_FLOAT, j);
              int ordinal = hostAllOrdinals.getAtIndex(ValueLayout.JAVA_INT, (int) pos);

              if (ordinal < 0) continue;
              segmentIndices[count] = (int) (pos / k);
              selectedOrdinals[count] = ordinal;
              selectedDistances[count] = dist;
              count++;
            }

            return new MultiSegmentSearchResults(
                count, segmentIndices, selectedOrdinals, selectedDistances);
          }
        } finally {
          if (filterHostArena != null) filterHostArena.close();
        }
      }
    }
  }

  /**
   * Populates a {@code cuvsFilter} MemorySegment for a MULTI_SEGMENT_BITSET filter using
   * pre-uploaded device buffers.
   */
  private static void buildCuvsFilterStruct(
      Arena arena,
      MemorySegment filterSeg,
      MemorySegment combinedBitsetHandle,
      MemorySegment segOffsetsHandle,
      long totalBits,
      int numSegments) {
    long[] bitsetShape = {(totalBits + 31) / 32};
    MemorySegment combinedBitsetTensor =
        prepareTensor(arena, combinedBitsetHandle, bitsetShape, kDLUInt(), 32, kDLCUDA());
    long[] offsetsShape = {numSegments};
    MemorySegment segOffsetsTensor =
        prepareTensor(arena, segOffsetsHandle, offsetsShape, kDLInt(), 64, kDLCUDA());

    // cuvsMultiSegmentBitsetFilter: {ptr combined_bitset, int64 total_bits, ptr segment_offsets}
    MemorySegment msbFilter = arena.allocate(24, 8);
    msbFilter.set(ValueLayout.JAVA_LONG, 0, combinedBitsetTensor.address());
    msbFilter.set(ValueLayout.JAVA_LONG, 8, totalBits);
    msbFilter.set(ValueLayout.JAVA_LONG, 16, segOffsetsTensor.address());

    cuvsFilter.type(filterSeg, 3 /* MULTI_SEGMENT_BITSET */);
    cuvsFilter.addr(filterSeg, msbFilter.address());
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
