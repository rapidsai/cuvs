/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.CagraIndexParams.CudaDataType;

public class CuVSIvfPqSearchParams {

  /** The number of clusters to search. */
  private final int nProbes;

  /**
   * Data type of look up table to be created dynamically at search time.
   *
   * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
   *
   * The use of low-precision types reduces the amount of shared memory required
   * at search time, so fast shared memory kernels can be used even for datasets
   * with large dimansionality. Note that the recall is slightly degraded when
   * low-precision type is selected.
   */
  private final CudaDataType lutDtype;

  /**
   * Storage data type for distance/similarity computed at search time.
   *
   * Possible values: [CUDA_R_16F, CUDA_R_32F]
   *
   * If the performance limiter at search time is device memory access, selecting
   * FP16 will improve performance slightly.
   */
  private final CudaDataType internalDistanceDtype;

  /**
   * Preferred fraction of SM's unified memory / L1 cache to be used as shared
   * memory.
   *
   * Possible values: [0.0 - 1.0] as a fraction of the
   * `sharedMemPerMultiprocessor`.
   *
   * One wants to increase the carveout to make sure a good GPU occupancy for the
   * main search kernel, but not to keep it too high to leave some memory to be
   * used as L1 cache. Note, this value is interpreted only as a hint. Moreover, a
   * GPU usually allows only a fixed set of cache configurations, so the provided
   * value is rounded up to the nearest configuration. Refer to the NVIDIA tuning
   * guide for the target GPU architecture.
   *
   * Note, this is a low-level tuning parameter that can have drastic negative
   * effects on the search performance if tweaked incorrectly.
   */
  private final double preferredShmemCarveout;

  private CuVSIvfPqSearchParams(
      int nProbes,
      CudaDataType lutDtype,
      CudaDataType internalDistanceDtype,
      double preferredShmemCarveout) {
    super();
    this.nProbes = nProbes;
    this.lutDtype = lutDtype;
    this.internalDistanceDtype = internalDistanceDtype;
    this.preferredShmemCarveout = preferredShmemCarveout;
  }

  /**
   * Gets the number of clusters to search
   *
   * @return the number of clusters to search
   */
  public int getnProbes() {
    return nProbes;
  }

  /**
   * Gets the data type of look up table to be created dynamically at search time
   *
   * @return the data type of look up table to be created dynamically at search
   *         time
   */
  public CudaDataType getLutDtype() {
    return lutDtype;
  }

  /**
   * Gets the storage data type for distance/similarity computed at search time
   *
   * @return the storage data type for distance/similarity computed at search time
   */
  public CudaDataType getInternalDistanceDtype() {
    return internalDistanceDtype;
  }

  /**
   * Gets the preferred fraction of SM's unified memory / L1 cache to be used as
   * shared memory
   *
   * @return the preferred fraction of SM's unified memory / L1 cache to be used
   *         as shared memory
   */
  public double getPreferredShmemCarveout() {
    return preferredShmemCarveout;
  }

  @Override
  public String toString() {
    return "CuVSIvfPqSearchParams [nProbes="
        + nProbes
        + ", lutDtype="
        + lutDtype
        + ", internalDistanceDtype="
        + internalDistanceDtype
        + ", preferredShmemCarveout="
        + preferredShmemCarveout
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link CuVSIvfPqSearchParams}.
   */
  public static class Builder {

    /** The number of clusters to search. */
    private int nProbes = 20;

    /**
     * Data type of look up table to be created dynamically at search time.
     *
     * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
     *
     * The use of low-precision types reduces the amount of shared memory required
     * at search time, so fast shared memory kernels can be used even for datasets
     * with large dimansionality. Note that the recall is slightly degraded when
     * low-precision type is selected.
     */
    private CudaDataType lutDtype = CudaDataType.CUDA_R_32F;

    /**
     * Storage data type for distance/similarity computed at search time.
     *
     * Possible values: [CUDA_R_16F, CUDA_R_32F]
     *
     * If the performance limiter at search time is device memory access, selecting
     * FP16 will improve performance slightly.
     */
    private CudaDataType internalDistanceDtype = CudaDataType.CUDA_R_32F;

    /**
     * Preferred fraction of SM's unified memory / L1 cache to be used as shared
     * memory.
     *
     * Possible values: [0.0 - 1.0] as a fraction of the
     * `sharedMemPerMultiprocessor`.
     *
     * One wants to increase the carveout to make sure a good GPU occupancy for the
     * main search kernel, but not to keep it too high to leave some memory to be
     * used as L1 cache. Note, this value is interpreted only as a hint. Moreover, a
     * GPU usually allows only a fixed set of cache configurations, so the provided
     * value is rounded up to the nearest configuration. Refer to the NVIDIA tuning
     * guide for the target GPU architecture.
     *
     * Note, this is a low-level tuning parameter that can have drastic negative
     * effects on the search performance if tweaked incorrectly.
     */
    private double preferredShmemCarveout = 1.0;

    public Builder() {}

    /**
     * Sets the number of clusters to search.
     *
     * @param nProbes the number of clusters to search
     * @return an instance of Builder
     */
    public Builder withNProbes(int nProbes) {
      this.nProbes = nProbes;
      return this;
    }

    /**
     * Sets the the data type of look up table to be created dynamically at search
     * time.
     *
     * @param lutDtype the data type of look up table to be created dynamically at
     *                 search time
     * @return an instance of Builder
     */
    public Builder withLutDtype(CudaDataType lutDtype) {
      this.lutDtype = lutDtype;
      return this;
    }

    /**
     * Sets the storage data type for distance/similarity computed at search time.
     *
     * @param internalDistanceDtype storage data type for distance/similarity
     *                              computed at search time
     * @return an instance of Builder
     */
    public Builder withInternalDistanceDtype(CudaDataType internalDistanceDtype) {
      this.internalDistanceDtype = internalDistanceDtype;
      return this;
    }

    /**
     * Sets the preferred fraction of SM's unified memory / L1 cache to be used as
     * shared memory.
     *
     * @param preferredShmemCarveout preferred fraction of SM's unified memory / L1
     *                               cache to be used as shared memory
     * @return an instance of Builder
     */
    public Builder withPreferredShmemCarveout(double preferredShmemCarveout) {
      this.preferredShmemCarveout = preferredShmemCarveout;
      return this;
    }

    /**
     * Builds an instance of {@link CuVSIvfPqSearchParams}.
     *
     * @return an instance of {@link CuVSIvfPqSearchParams}
     */
    public CuVSIvfPqSearchParams build() {
      return new CuVSIvfPqSearchParams(
          nProbes, lutDtype, internalDistanceDtype, preferredShmemCarveout);
    }
  }
}
