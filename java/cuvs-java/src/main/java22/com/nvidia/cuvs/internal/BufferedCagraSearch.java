/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.CagraQuery;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Internal interface implemented by CAGRA index classes that support writing
 * search results directly into a caller-owned device buffer without syncing
 * the stream or copying results to host.
 *
 * <p>Used by {@link com.nvidia.cuvs.MultiSegmentCagraSearch} to queue all
 * per-segment searches before running a single GPU-side top-k reduction.
 */
public interface BufferedCagraSearch {

  /**
   * Runs CAGRA search and writes results into a slice of caller-owned device
   * buffers without copying results to host or syncing the stream.
   *
   * <p>Results are written at element offset {@code segmentIdx * query.getTopK()}
   * in each buffer:
   * <ul>
   *   <li>{@code globalNeighborsDP}: uint32 ordinals</li>
   *   <li>{@code globalDistancesDP}: float32 distances</li>
   * </ul>
   *
   * <p>The search is submitted to {@code segmentStream}. The caller is responsible for
   * synchronizing that stream (e.g. via a CUDA event) before consuming the output buffers.
   *
   * @param query             query with vectors, topK, search params, optional prefilter
   * @param globalNeighborsDP device pointer to the shared uint32 neighbors buffer
   * @param globalDistancesDP device pointer to the shared float32 distances buffer
   * @param segmentIdx        zero-based segment index; determines the write offset
   * @param segmentCuvsRes    {@code cuvsResources_t} handle whose CUDA stream receives the kernel
   * @param segmentStream     CUDA stream corresponding to {@code segmentCuvsRes}; passed explicitly
   *                          to avoid a redundant {@code cuvsStreamGet} call inside the method
   * @param searchParams      pre-built {@code cuvsCagraSearchParams} struct; shared across all
   *                          segments to avoid repeated allocation and population
   * @param arena             shared scratch arena for per-call CPU-side allocations (tensor
   *                          descriptors, filter struct); must remain open until after this call
   *                          returns, and the GPU kernel has launched
   */
  void searchIntoBuffer(
      CagraQuery query,
      MemorySegment globalNeighborsDP,
      MemorySegment globalDistancesDP,
      int segmentIdx,
      long segmentCuvsRes,
      MemorySegment segmentStream,
      MemorySegment searchParams,
      Arena arena)
      throws Throwable;

  /**
   * Returns the raw {@code cuvsCagraIndex_t} handle as a {@link MemorySegment}.
   * Used by {@link com.nvidia.cuvs.MultiSegmentCagraSearch} to build the index pointer array
   * for {@code cuvsCagraSearchMultiSegment}.
   */
  MemorySegment getIndexHandle();
}
