/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.CagraQuery;
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
   * <p>The caller must synchronize the stream after all segments have been searched.
   *
   * @param query             query with vectors, topK, search params, optional prefilter
   * @param globalNeighborsDP device pointer to the shared uint32 neighbors buffer
   * @param globalDistancesDP device pointer to the shared float32 distances buffer
   * @param segmentIdx        zero-based segment index; determines the write offset
   */
  void searchIntoBuffer(
      CagraQuery query,
      MemorySegment globalNeighborsDP,
      MemorySegment globalDistancesDP,
      int segmentIdx)
      throws Throwable;
}
