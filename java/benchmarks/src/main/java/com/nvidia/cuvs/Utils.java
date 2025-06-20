package com.nvidia.cuvs;

import java.lang.foreign.*;
import java.util.Random;

class Utils {
    private static final Random random = new Random();

    static float[][] createSampleData(int size, int dimensions) {
        var array = new float[size][dimensions];
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                array[i][j] = random.nextFloat();
            }
        }
        return array;
    }

    static MemorySegment createSampleDataSegment(Arena arena, float[][] array, int size, int dimensions) {
        final ValueLayout.OfFloat C_FLOAT = (ValueLayout.OfFloat) Linker.nativeLinker().canonicalLayouts().get("float");

        MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout((long)size * dimensions, C_FLOAT);

        var segment = arena.allocate(dataMemoryLayout);
        for (int i = 0; i < size; ++i) {
            var vector = array[i];
            MemorySegment.copy(vector, 0, segment, C_FLOAT, (i * dimensions * C_FLOAT.byteSize()), dimensions);
        }
        return segment;
    }
}
