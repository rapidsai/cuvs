package com.nvidia.cuvs.internal;

import java.lang.foreign.MemorySegment;

interface MemorySegmentProvider {
    MemorySegment asMemorySegment();
}
