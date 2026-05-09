/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Holds the decoded results of a multi-segment GPU search.
 *
 * <p>Each entry {@code i} in [0, {@link #count}) identifies:
 * <ul>
 *   <li>which input segment the result came from ({@link #getSegmentIndex(int)})</li>
 *   <li>the local vector ordinal within that segment ({@link #getOrdinal(int)})</li>
 *   <li>the raw CAGRA distance ({@link #getDistance(int)})</li>
 * </ul>
 *
 * <p>The caller is responsible for mapping ordinals to Lucene doc IDs using the
 * segment-specific {@code ordToDoc} function and adding {@code docBase}.
 *
 * @since 25.10
 */
public class MultiSegmentSearchResults {

  private final int count;
  private final int[] segmentIndices;
  private final int[] ordinals;
  private final float[] distances;

  MultiSegmentSearchResults(int count, int[] segmentIndices, int[] ordinals, float[] distances) {
    this.count = count;
    this.segmentIndices = segmentIndices;
    this.ordinals = ordinals;
    this.distances = distances;
  }

  /** Number of valid results (may be less than k if fewer candidates exist). */
  public int count() {
    return count;
  }

  /** Index into the original segment list for result {@code i}. */
  public int getSegmentIndex(int i) {
    return segmentIndices[i];
  }

  /** Local vector ordinal within the segment for result {@code i}. */
  public int getOrdinal(int i) {
    return ordinals[i];
  }

  /** Raw CAGRA distance for result {@code i} (before score normalization). */
  public float getDistance(int i) {
    return distances[i];
  }
}
