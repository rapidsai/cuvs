/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.DEVICE_TO_HOST;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpyAsync;
import static com.nvidia.cuvs.internal.common.Util.getStream;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraSearchMultiSegment;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLFloat;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLUInt;

import com.nvidia.cuvs.internal.BufferedCagraSearch;
import com.nvidia.cuvs.internal.CuVSMatrixInternal;
import com.nvidia.cuvs.internal.CuVSParamsHelper;
import com.nvidia.cuvs.internal.SelectKHelper;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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
   * @param resources  shared {@link CuVSResources} handle; all queries must use the same instance
   * @param indices    one {@link CagraIndex} per segment, in segment order; each must implement
   *                   {@link BufferedCagraSearch} (all built-in implementations do)
   * @param queries    one {@link CagraQuery} per segment (same topK for all); each query encodes
   *                   the target vector, search parameters, and optional prefilter for that segment
   * @param k          number of global nearest neighbors to return
   * @return decoded search results with per-result (segmentIndex, ordinal, distance)
   * @throws IllegalArgumentException if {@code indices} and {@code queries} differ in size, or if
   *                                  any index does not support buffered search
   */
  public static MultiSegmentSearchResults search(
      CuVSResources resources, List<CagraIndex> indices, List<CagraQuery> queries, int k)
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

    // Validate that all indices support buffered search.
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
    long neighborsBytes = totalCandidates * Integer.BYTES; // uint32 per ordinal
    long distancesBytes = totalCandidates * Float.BYTES; // float32 per distance
    long outIdxBytes = (long) k * Long.BYTES; // int64 positions from select_k
    long outValBytes = (long) k * Float.BYTES;

    CagraSearchParams searchParameters = queries.get(0).getCagraSearchParameters();

    try (var resourcesAccessor = resources.access()) {
      long cuvsRes = resourcesAccessor.handle();
      var cuvsStream = getStream(cuvsRes);

      try (var globalNeighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
          var globalDistancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
          var outIdxDP = allocateRMMSegment(cuvsRes, outIdxBytes);
          var outValDP = allocateRMMSegment(cuvsRes, outValBytes)) {

        // --- Phase 1: call cuvsCagraSearchMultiSegment ---
        // Single kernel launch covers all segments; results land in globalNeighborsDP /
        // globalDistancesDP on the same CUDA stream, so SelectK below sees them via ordering.
        try (var arena = Arena.ofConfined()) {
          MemorySegment sp = CuVSParamsHelper.buildCagraSearchParams(arena, searchParameters);

          // Build C arrays: cuvsCagraIndex_t* indices, DLManagedTensor** q/n/d
          MemorySegment indexArray = arena.allocate(ValueLayout.ADDRESS, numSegments);
          MemorySegment queriesArray = arena.allocate(ValueLayout.ADDRESS, numSegments);
          MemorySegment neighborsArray = arena.allocate(ValueLayout.ADDRESS, numSegments);
          MemorySegment distancesArray = arena.allocate(ValueLayout.ADDRESS, numSegments);

          long[] segShape = {1, k};
          for (int i = 0; i < numSegments; i++) {
            // Index handle
            indexArray.setAtIndex(ValueLayout.ADDRESS, i, buffered[i].getIndexHandle());

            // Query DLTensor
            var queryVectors = (CuVSMatrixInternal) queries.get(i).getQueryVectors();
            queriesArray.setAtIndex(ValueLayout.ADDRESS, i, queryVectors.toTensor(arena));

            // Neighbors DLTensor — slice of global buffer
            long nByteOffset = (long) i * k * Integer.BYTES;
            MemorySegment nSlice =
                MemorySegment.ofAddress(globalNeighborsDP.handle().address() + nByteOffset);
            neighborsArray.setAtIndex(
                ValueLayout.ADDRESS,
                i,
                prepareTensor(arena, nSlice, segShape, kDLUInt(), 32, kDLCUDA()));

            // Distances DLTensor — slice of global buffer
            long dByteOffset = (long) i * k * Float.BYTES;
            MemorySegment dSlice =
                MemorySegment.ofAddress(globalDistancesDP.handle().address() + dByteOffset);
            distancesArray.setAtIndex(
                ValueLayout.ADDRESS,
                i,
                prepareTensor(arena, dSlice, segShape, kDLFloat(), 32, kDLCUDA()));
          }

          checkCuVSError(
              cuvsCagraSearchMultiSegment(
                  cuvsRes,
                  sp,
                  numSegments,
                  indexArray,
                  queriesArray,
                  neighborsArray,
                  distancesArray),
              "cuvsCagraSearchMultiSegment");
        }

        // --- Phase 2: select global top-k on GPU ---
        SelectKHelper.selectK(
            cuvsRes,
            globalDistancesDP.handle(),
            totalCandidates,
            outValDP.handle(),
            outIdxDP.handle(),
            k);

        // No stream sync needed here: the D2H copies below are enqueued on the same cuvsStream,
        // so CUDA stream ordering guarantees selectK completes before the copies begin.

        // --- Phase 3: single device-to-host copy for all three arrays ---
        // Layout (in order of decreasing alignment): int64 outIdx | float32 outVal | uint32
        // ordinals
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

          // --- Phase 4: decode results ---
          int[] segmentIndices = new int[k];
          int[] selectedOrdinals = new int[k];
          float[] selectedDistances = new float[k];
          int count = 0;

          for (int j = 0; j < k; j++) {
            long pos = hostOutIdx.getAtIndex(ValueLayout.JAVA_LONG, j);
            float dist = hostOutVal.getAtIndex(ValueLayout.JAVA_FLOAT, j);
            int ordinal = hostAllOrdinals.getAtIndex(ValueLayout.JAVA_INT, (int) pos);

            if (ordinal < 0) {
              // CAGRA uses negative sentinel values for unfilled slots
              continue;
            }
            segmentIndices[count] = (int) (pos / k);
            selectedOrdinals[count] = ordinal;
            selectedDistances[count] = dist;
            count++;
          }

          return new MultiSegmentSearchResults(
              count, segmentIndices, selectedOrdinals, selectedDistances);
        }
      }
    }
  }
}
