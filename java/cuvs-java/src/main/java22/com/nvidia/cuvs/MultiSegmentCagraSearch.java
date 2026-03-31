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
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;

import com.nvidia.cuvs.internal.BufferedCagraSearch;
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
 *   <li>For each segment, call {@link BufferedCagraSearch#searchIntoBuffer} to queue the CAGRA
 *       search kernel; results are written into the segment's slice with no stream sync between
 *       segments.</li>
 *   <li>Sync the stream once after all segment searches are queued.</li>
 *   <li>Call {@code cuvsSelectK} to find the global top-k smallest distances across all
 *       {@code numSegments × k} candidates entirely on GPU.</li>
 *   <li>Sync the stream again.</li>
 *   <li>Copy the three result arrays to host in a single pass:
 *       k selected distances, k flat-array positions, and all {@code numSegments × k} ordinals.</li>
 *   <li>Decode each result: {@code segment = position / k}, {@code ordinal = ordinals[position]}.</li>
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

    try (var resourcesAccessor = resources.access()) {
      long cuvsRes = resourcesAccessor.handle();
      var cuvsStream = getStream(cuvsRes);

      try (var globalNeighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
          var globalDistancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
          var outIdxDP = allocateRMMSegment(cuvsRes, outIdxBytes);
          var outValDP = allocateRMMSegment(cuvsRes, outValBytes)) {

        // --- Phase 1: queue all per-segment CAGRA searches ---
        for (int i = 0; i < numSegments; i++) {
          buffered[i].searchIntoBuffer(
              queries.get(i), globalNeighborsDP.handle(), globalDistancesDP.handle(), i);
        }

        // --- Phase 2: sync once, then select global top-k on GPU ---
        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync before selectK");

        SelectKHelper.selectK(
            cuvsRes,
            globalDistancesDP.handle(),
            totalCandidates,
            outValDP.handle(),
            outIdxDP.handle(),
            k);

        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync after selectK");

        // --- Phase 3: single device-to-host copy for all three arrays ---
        try (var arena = Arena.ofConfined()) {
          MemorySegment hostOutIdx = arena.allocate(outIdxBytes);
          MemorySegment hostOutVal = arena.allocate(outValBytes);
          MemorySegment hostAllOrdinals = arena.allocate(neighborsBytes);

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
