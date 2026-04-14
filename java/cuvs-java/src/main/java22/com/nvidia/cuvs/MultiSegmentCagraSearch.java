/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.DEVICE_TO_HOST;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpyAsync;
import static com.nvidia.cuvs.internal.common.Util.getStream;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaEventRecord;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaStreamWaitEvent;

import com.nvidia.cuvs.internal.BufferedCagraSearch;
import com.nvidia.cuvs.internal.CuVSParamsHelper;
import com.nvidia.cuvs.internal.CuVSResourcesImpl;
import com.nvidia.cuvs.internal.CudaStreamPool;
import com.nvidia.cuvs.internal.SelectKHelper;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Performs a single-query approximate nearest neighbor search across multiple CAGRA index segments
 * using a shared GPU buffer and a fixed-size CUDA stream pool, eliminating per-segment
 * device-to-host copies.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Allocate two global device buffers sized {@code numSegments × k}:
 *       one for uint32 neighbor ordinals and one for float32 distances.</li>
 *   <li>Assign each segment a slot from the {@link CudaStreamPool} via round-robin. Segments on
 *       different slots run in parallel on separate CUDA streams.</li>
 *   <li>For each segment, call {@link BufferedCagraSearch#searchIntoBuffer} to queue the CAGRA
 *       search kernel. In non-persistent mode this enqueues asynchronously on the slot's CUDA
 *       stream. In persistent mode each call blocks on CPU until the GPU worker signals completion;
 *       all segments are submitted concurrently via {@link #ASYNC_SEARCH_POOL} so the GPU can
 *       execute multiple segment jobs in parallel (bounded by {@code worker_queue_size}).</li>
 *   <li>Record a CUDA event on each slot's stream; make the main stream wait on all events.</li>
 *   <li>Call {@code cuvsSelectK} on the main stream to find the global top-k entirely on GPU.</li>
 *   <li>Sync the main stream.</li>
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
   * Thread pool used to submit persistent-mode segment searches concurrently.
   *
   * <p>In persistent mode, {@link BufferedCagraSearch#searchIntoBuffer} blocks on the CPU until
   * the GPU signals completion via a system-scope atomic. Running each segment in its own thread
   * allows the persistent kernel's job queue to hold all N segment jobs simultaneously, so GPU
   * workers can execute them in parallel (bounded by {@code worker_queue_size}).
   */
  private static final ExecutorService ASYNC_SEARCH_POOL =
      Executors.newCachedThreadPool(
          r -> {
            Thread t = new Thread(r, "cuvs-segment-search");
            t.setDaemon(true);
            return t;
          });

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

    // Assign a pool slot to each segment via round-robin.
    CudaStreamPool pool = CuVSResourcesImpl.getStreamPool(resources);
    int startSlot = pool.nextSlot(numSegments);
    int[] slots = new int[numSegments];
    for (int i = 0; i < numSegments; i++) {
      slots[i] = Math.floorMod(startSlot + i, pool.size());
    }

    try (var resourcesAccessor = resources.access()) {
      long cuvsRes = resourcesAccessor.handle();
      var cuvsStream = getStream(cuvsRes);

      try (var globalNeighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
          var globalDistancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
          var outIdxDP = allocateRMMSegment(cuvsRes, outIdxBytes);
          var outValDP = allocateRMMSegment(cuvsRes, outValBytes)) {

        // --- Phase 1: queue all per-segment CAGRA searches ---
        CagraSearchParams searchParameters = queries.get(0).getCagraSearchParameters();
        if (searchParameters.isPersistent()) {
          // Persistent mode: searchIntoBuffer blocks on CPU (via system-scope atomic spin) until
          // the GPU signals completion. Submit one task per pool slot in parallel so the GPU can
          // work on multiple segment jobs concurrently, bounded by worker_queue_size.
          //
          // Segments are grouped by slot: if numSegments > pool.size(), multiple segments share a
          // slot and must be serialized within that slot's task — each cuvsResources_t handle is
          // not thread-safe for concurrent access (the descriptor_cache inside is not guarded).
          // Parallelism = min(numSegments, pool.size()).
          int poolSize = pool.size();
          // Collect segment indices per slot. Size: poolSize, each entry may have 0..n indices.
          @SuppressWarnings("unchecked")
          List<Integer>[] segsBySlot = new List[poolSize];
          for (int slot = 0; slot < poolSize; slot++) {
            segsBySlot[slot] = new ArrayList<>();
          }
          for (int i = 0; i < numSegments; i++) {
            segsBySlot[slots[i]].add(i);
          }
          // Submit one task per occupied slot.
          List<Future<Void>> futures = new ArrayList<>(poolSize);
          for (int slot = 0; slot < poolSize; slot++) {
            if (segsBySlot[slot].isEmpty()) continue;
            final int taskSlot = slot;
            final List<Integer> taskSegs = segsBySlot[slot];
            futures.add(
                ASYNC_SEARCH_POOL.submit(
                    (Callable<Void>)
                        () -> {
                          try (var threadArena = Arena.ofConfined()) {
                            MemorySegment sp =
                                CuVSParamsHelper.buildCagraSearchParams(
                                    threadArena, searchParameters);
                            for (int segIdx : taskSegs) {
                              buffered[segIdx].searchIntoBuffer(
                                  queries.get(segIdx),
                                  globalNeighborsDP.handle(),
                                  globalDistancesDP.handle(),
                                  segIdx,
                                  pool.resources(taskSlot),
                                  pool.stream(taskSlot),
                                  sp,
                                  threadArena);
                            }
                          } catch (Exception e) {
                            throw e;
                          } catch (Throwable t) {
                            throw new RuntimeException(t);
                          }
                          return null;
                        }));
          }
          for (Future<Void> f : futures) {
            try {
              f.get();
            } catch (ExecutionException e) {
              throw e.getCause();
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
              throw e;
            }
          }
        } else {
          // Non-persistent: each cuvsCagraSearch enqueues a CUDA kernel asynchronously and
          // returns immediately; segments execute in parallel on their respective CUDA streams.
          // A shared arena covers all per-call CPU allocations; it is closed once all launches
          // have been enqueued.
          try (var segArena = Arena.ofConfined()) {
            MemorySegment searchParams =
                CuVSParamsHelper.buildCagraSearchParams(segArena, searchParameters);
            for (int i = 0; i < numSegments; i++) {
              buffered[i].searchIntoBuffer(
                  queries.get(i),
                  globalNeighborsDP.handle(),
                  globalDistancesDP.handle(),
                  i,
                  pool.resources(slots[i]),
                  pool.stream(slots[i]),
                  searchParams,
                  segArena);
            }
          }
        }

        // --- Phase 2: event-based sync — make main stream wait for all segment streams ---
        // Record one event per distinct slot (on the last kernel submitted to that slot);
        // this is O(pool.size()) API calls instead of O(numSegments).
        // Pool events are pre-allocated and reused across calls to avoid create/destroy overhead.
        int[] lastSegmentForSlot = new int[pool.size()];
        Arrays.fill(lastSegmentForSlot, -1);
        for (int i = 0; i < numSegments; i++) {
          lastSegmentForSlot[slots[i]] = i;
        }
        for (int slot = 0; slot < pool.size(); slot++) {
          if (lastSegmentForSlot[slot] >= 0) {
            checkCudaError(cudaEventRecord(pool.event(slot), pool.stream(slot)), "cudaEventRecord");
            checkCudaError(
                cudaStreamWaitEvent(cuvsStream, pool.event(slot), 0), "cudaStreamWaitEvent");
          }
        }

        // --- Phase 3: select global top-k on GPU (after all segment searches complete) ---
        SelectKHelper.selectK(
            cuvsRes,
            globalDistancesDP.handle(),
            totalCandidates,
            outValDP.handle(),
            outIdxDP.handle(),
            k);

        // No stream sync needed here: the D2H copies below are enqueued on the same cuvsStream,
        // so CUDA stream ordering guarantees selectK completes before the copies begin.

        // --- Phase 4: single device-to-host copy for all three arrays ---
        // Allocate one contiguous host buffer and slice into three typed views.
        // Layout (in order of decreasing alignment): int64 outIdx | float32 outVal | uint32
        // ordinals
        // outIdxBytes is a multiple of Long.BYTES, so each slice is naturally aligned.
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

          // --- Phase 5: decode results ---
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
