package com.nvidia.cuvs;

/**
 * Contains GPU information
 */
public class GPUInfo {
  private final int gpuId;
  private final String name;
  private final long freeMemory;
  private final long totalMemory;
  private final float computeCapability;

  /**
   * Constructor for GPUInfo
   *
   * @param gpuId             id of the GPU starting from 0
   * @param name              ASCII string identifying device
   * @param freeMemory        returned free memory in bytes
   * @param totalMemory       returned total memory in bytes
   * @param computeCapability the compute capability of the device
   */
  public GPUInfo(int gpuId, String name, long freeMemory, long totalMemory, float computeCapability) {
    super();
    this.gpuId = gpuId;
    this.name = name;
    this.freeMemory = freeMemory;
    this.totalMemory = totalMemory;
    this.computeCapability = computeCapability;
  }

  public int getGpuId() {
    return gpuId;
  }

  public String getName() {
    return name;
  }

  public long getFreeMemory() {
    return freeMemory;
  }

  public long getTotalMemory() {
    return totalMemory;
  }

  public float getComputeCapability() {
    return computeCapability;
  }

  @Override
  public String toString() {
    return "GPUInfo [gpuId=" + gpuId + ", name=" + name + ", freeMemory=" + freeMemory + ", totalMemory=" + totalMemory
        + ", computeCapability=" + computeCapability + "]";
  }
}
