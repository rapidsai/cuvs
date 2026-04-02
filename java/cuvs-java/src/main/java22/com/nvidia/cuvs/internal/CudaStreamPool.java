/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaEventCreateWithFlags;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaEventDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResourcesCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResourcesDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResources_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSet;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaEventDisableTiming;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaEvent_t;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaStreamCreateWithFlags;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaStreamDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaStreamNonBlocking;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaStream_t;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A fixed-size pool of CUDA streams used by {@link com.nvidia.cuvs.MultiSegmentCagraSearch}
 * to run per-segment CAGRA searches in parallel.
 *
 * <p>Each pool slot owns one {@code cuvsResources_t} handle backed by a dedicated non-blocking
 * CUDA stream. Callers assign segments to slots via round-robin using {@link #nextSlot}; the
 * GPU executes searches on different slots concurrently, then synchronizes via CUDA events before
 * the global {@code cuvsSelectK} call.
 *
 * <h3>Lifecycle</h3>
 * <p>One pool is owned by each {@code CuVSResourcesImpl} instance and closed when that instance
 * is closed. This gives each thread its own independent set of streams and events, eliminating
 * races when multiple threads perform concurrent multi-segment searches.
 *
 * <h3>Configuration</h3>
 * <p>Pool size defaults to {@value #DEFAULT_SIZE} and can be overridden via the system property
 * {@value #SIZE_PROPERTY}.
 */
public final class CudaStreamPool implements AutoCloseable {

  /** Default number of streams in the pool. */
  public static final int DEFAULT_SIZE = 8;

  /** System property name for overriding the pool size. */
  public static final String SIZE_PROPERTY = "com.nvidia.cuvs.streamPoolSize";

  /** Round-robin counter; advanced by {@link #nextSlot(int)} on each search call. */
  private int slotCounter;

  private final long[] resources; // cuvsResources_t handles
  private final MemorySegment[] streams; // cudaStream_t handle values
  private final MemorySegment[] events; // pre-allocated cudaEvent_t handles, one per slot
  private final int size;

  CudaStreamPool(int size) {
    this.size = size;
    this.resources = new long[size];
    this.streams = new MemorySegment[size];
    this.events = new MemorySegment[size];
    try (var arena = Arena.ofConfined()) {
      for (int i = 0; i < size; i++) {
        // Create a non-blocking CUDA stream (avoids implicit sync with the default stream).
        var pStream = arena.allocate(cudaStream_t);
        checkCudaError(
            cudaStreamCreateWithFlags(pStream, cudaStreamNonBlocking()), "cudaStreamCreate");
        streams[i] = pStream.get(cudaStream_t, 0);

        // Create a cuvsResources_t and assign the new stream to it.
        var pRes = arena.allocate(cuvsResources_t);
        checkCuVSError(cuvsResourcesCreate(pRes), "cuvsResourcesCreate");
        resources[i] = pRes.get(cuvsResources_t, 0);
        checkCuVSError(cuvsStreamSet(resources[i], streams[i]), "cuvsStreamSet");

        // Pre-allocate one reusable event per slot (disable timing to avoid overhead).
        var pEvent = arena.allocate(cudaEvent_t);
        checkCudaError(
            cudaEventCreateWithFlags(pEvent, cudaEventDisableTiming()), "cudaEventCreate");
        events[i] = pEvent.get(cudaEvent_t, 0);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Per-slot accessors
  // -------------------------------------------------------------------------

  /** Returns the {@code cuvsResources_t} handle for the given slot. */
  public long resources(int slot) {
    return resources[slot];
  }

  /** Returns the CUDA stream handle for the given slot. */
  public MemorySegment stream(int slot) {
    return streams[slot];
  }

  /** Returns the pre-allocated CUDA event handle for the given slot. */
  public MemorySegment event(int slot) {
    return events[slot];
  }

  /** Returns the number of slots in this pool. */
  public int size() {
    return size;
  }

  /**
   * Advances the round-robin counter by {@code count} and returns the starting slot index for
   * this call. Slot indices are wrapped modulo {@link #size()}.
   */
  public int nextSlot(int count) {
    int start = slotCounter;
    slotCounter += count;
    return start;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  @Override
  public void close() {
    for (int i = 0; i < size; i++) {
      checkCudaError(cudaEventDestroy(events[i]), "cudaEventDestroy");
      cuvsResourcesDestroy(resources[i]);
      checkCudaError(cudaStreamDestroy(streams[i]), "cudaStreamDestroy");
    }
  }
}
