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

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.foreign.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.UUID;

@Fork(value = 1, warmups = 0)
@State(Scope.Benchmark)
public class CagraIndexBenchmarks {

    @Param({ "1024" })
    private int dims;

    @Param({ "100" })
    private int size;

    private float[][] arrayDataset;

    private static final Random random = new Random();

    private static float[][] createSampleData(int size, int dimensions) {
        var array = new float[size][dimensions];
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                array[i][j] = random.nextFloat();
            }
        }
        return array;
    }

    @Setup
    public void initialize() {
        arrayDataset = createSampleData(size, dims);
    }

    @Benchmark
    public void testIndexingAndSerializeToFile() throws Throwable {
        try (CuVSResources resources = CuVSResources.create()) {
            // Configure index parameters
            CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(1)
                .withIntermediateGraphDegree(2)
                .withNumWriterThreads(32)
                .withMetric(CagraIndexParams.CuvsDistanceType.L2Expanded)
                .build();

            // Create the index with the dataset
            CagraIndex index = CagraIndex.newBuilder(resources)
                .withDataset(arrayDataset)
                .withIndexParams(indexParams)
                .build();

            // Saving the index on to the disk.
            var indexFilePath = Path.of(UUID.randomUUID() + ".cag");
            try (var outputStream = Files.newOutputStream(indexFilePath)) {
                index.serialize(outputStream);
            }

            // Cleanup
            Files.deleteIfExists(indexFilePath);
            index.destroyIndex();
        }
    }

    @Benchmark
    public void testIndexingFromHeap(Blackhole blackhole) throws Throwable {
        try (CuVSResources resources = CuVSResources.create()) {

            // Configure index parameters
            CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(1)
                .withIntermediateGraphDegree(2)
                .withNumWriterThreads(32)
                .withMetric(CagraIndexParams.CuvsDistanceType.L2Expanded)
                .build();

            // Create the index with the dataset
            CagraIndex index = CagraIndex.newBuilder(resources)
                .withDataset(arrayDataset)
                .withIndexParams(indexParams)
                .build();
            blackhole.consume(index);
        }
    }
}
