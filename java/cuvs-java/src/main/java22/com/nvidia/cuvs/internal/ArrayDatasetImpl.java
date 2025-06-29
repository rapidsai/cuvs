package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.internal.common.Util;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

public class ArrayDatasetImpl implements Dataset, MemorySegmentProvider {
    private final float[][] vectors;

    public ArrayDatasetImpl(float[][] vectors) {
        this.vectors = Objects.requireNonNull(vectors);
        if (vectors.length == 0) {
            throw new IllegalArgumentException("vectors should not be empty");
        }
    }

    @Override
    public void addVector(float[] vector) {

    }

    @Override
    public int size() {
        return vectors.length;
    }

    @Override
    public int dimensions() {
        return vectors[0].length;
    }

    @Override
    public void close() { }

    @Override
    public MemorySegment asMemorySegment(Arena arena) {
        return Util.buildMemorySegment(arena, vectors);
    }
}
