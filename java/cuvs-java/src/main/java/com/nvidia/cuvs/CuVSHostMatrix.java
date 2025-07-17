package com.nvidia.cuvs;

/**
 * A Dataset implementation backed by host (CPU) memory.
 */
public interface CuVSHostMatrix extends CuVSMatrix {
    int get(int row, int col);
}
