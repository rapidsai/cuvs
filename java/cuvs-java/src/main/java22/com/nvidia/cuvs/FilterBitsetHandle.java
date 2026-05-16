/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.HOST_TO_DEVICE;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpyAsync;
import static com.nvidia.cuvs.internal.common.Util.getStream;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;

import com.nvidia.cuvs.internal.common.CloseableRMMAllocation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Holds a precomputed multi-segment filter bitset and manages its device-memory lifecycle.
 *
 * <h3>Two-level caching</h3>
 * <ul>
 *   <li><strong>Host level</strong> – the packed {@code long[]} arrays are owned by this object
 *       and shared safely across threads (immutable after construction).</li>
 *   <li><strong>Device level</strong> – a single shared device allocation is uploaded once on
 *       first use (lazy, double-checked locking) and reused by all threads thereafter.</li>
 * </ul>
 *
 * <p>Callers must call {@link #close()} when the handle is evicted from their host-level cache.
 *
 * @since 25.10
 */
public final class FilterBitsetHandle implements AutoCloseable {

  /** Device-side allocation pair, shared across all threads. */
  static final class DeviceData {
    final CloseableRMMAllocation combinedBitsetDP;
    final CloseableRMMAllocation segOffsetsDP;
    final long totalBits;
    final int numSegments;

    DeviceData(
        CloseableRMMAllocation combinedBitsetDP,
        CloseableRMMAllocation segOffsetsDP,
        long totalBits,
        int numSegments) {
      this.combinedBitsetDP = combinedBitsetDP;
      this.segOffsetsDP = segOffsetsDP;
      this.totalBits = totalBits;
      this.numSegments = numSegments;
    }

    void close() {
      try {
        combinedBitsetDP.close();
      } catch (Exception ignored) {
      }
      try {
        segOffsetsDP.close();
      } catch (Exception ignored) {
      }
    }
  }

  // Host-side immutable data.
  final long[] combinedLongs;
  final long[] segBitOffsets;
  final long totalBits;
  final int numSegments;

  // Shared device allocation — uploaded once, visible to all threads via volatile.
  private volatile DeviceData sharedDeviceData;
  private final Object uploadLock = new Object();

  private volatile boolean closed = false;

  /**
   * Creates a handle from pre-packed host arrays.
   *
   * @param combinedLongs  packed bitset words for all segments concatenated (64-bit aligned)
   * @param segBitOffsets  per-segment bit offsets into {@code combinedLongs}
   * @param totalBits      total number of logical bits in {@code combinedLongs}
   */
  public FilterBitsetHandle(long[] combinedLongs, long[] segBitOffsets, long totalBits) {
    this.combinedLongs = combinedLongs;
    this.segBitOffsets = segBitOffsets;
    this.totalBits = totalBits;
    this.numSegments = segBitOffsets.length;
  }

  /**
   * Returns the shared device allocation for this filter, uploading on first call.
   *
   * <p>The upload uses stream-ordered {@code cudaMemcpyAsync} followed by
   * {@code cuvsStreamSync}, so no other stream is serialized.
   *
   * @param cuvsRes the native cuvsResources handle for the calling thread
   * @return shared device data (valid until {@link #close()} is called)
   */
  DeviceData getOrUpload(long cuvsRes) {
    if (closed) throw new IllegalStateException("FilterBitsetHandle has been closed");
    DeviceData data = sharedDeviceData;
    if (data != null) return data;
    synchronized (uploadLock) {
      data = sharedDeviceData;
      if (data != null) return data;
      data = upload(cuvsRes);
      sharedDeviceData = data; // volatile write: happens-before all subsequent reads
    }
    return data;
  }

  private DeviceData upload(long cuvsRes) {
    long combinedBitsetBytes = (long) combinedLongs.length * Long.BYTES;
    long segOffsetsBytes = (long) segBitOffsets.length * Long.BYTES;

    CloseableRMMAllocation combinedBitsetDP = allocateRMMSegment(cuvsRes, combinedBitsetBytes);
    CloseableRMMAllocation segOffsetsDP = allocateRMMSegment(cuvsRes, segOffsetsBytes);

    var stream = getStream(cuvsRes);
    // Host arenas must outlive the stream sync that confirms the H2D copies.
    try (var arena = Arena.ofConfined()) {
      MemorySegment hostBitset = arena.allocate(combinedBitsetBytes, Long.BYTES);
      MemorySegment.copy(
          combinedLongs, 0, hostBitset, ValueLayout.JAVA_LONG, 0, combinedLongs.length);
      cudaMemcpyAsync(
          combinedBitsetDP.handle(), hostBitset, combinedBitsetBytes, HOST_TO_DEVICE, stream);

      MemorySegment hostOffsets = arena.allocate(segOffsetsBytes, Long.BYTES);
      MemorySegment.copy(
          segBitOffsets, 0, hostOffsets, ValueLayout.JAVA_LONG, 0, segBitOffsets.length);
      cudaMemcpyAsync(segOffsetsDP.handle(), hostOffsets, segOffsetsBytes, HOST_TO_DEVICE, stream);

      checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync in FilterBitsetHandle.upload");
    }
    // Stream sync has returned — device memory is fully populated.
    return new DeviceData(combinedBitsetDP, segOffsetsDP, totalBits, numSegments);
  }

  /** Marks this handle closed and releases the shared device allocation. */
  @Override
  public void close() {
    closed = true;
    DeviceData data;
    synchronized (uploadLock) {
      data = sharedDeviceData;
      sharedDeviceData = null;
    }
    if (data != null) data.close();
  }
}
