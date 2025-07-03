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
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.foreign.*;
import java.util.concurrent.TimeUnit;

import static com.nvidia.cuvs.Utils.createSampleData;
import static com.nvidia.cuvs.Utils.createSampleDataSegment;

@Fork(value = 1, warmups = 0)
@State(Scope.Benchmark)
public class CagraIndexingBenchmarks {

    @Param({ "1024" })
    private int dims;

    @Param({ "100" })
    private int size;

    private float[][] arrayDataset;

    private Arena arena;

    private MemorySegment memorySegmentDataset;

    @Setup
    public void initialize() {
        arena = Arena.ofShared();
        arrayDataset = createSampleData(size, dims);
        memorySegmentDataset = createSampleDataSegment(arena, arrayDataset, size, dims);
    }

    @TearDown
    public void cleanUp() {
        if (arena != null) {
            arena.close();
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

    @Benchmark
    public void testIndexingFromMemorySegment(Blackhole blackhole) throws Throwable {
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
                .withDataset(Dataset.ofMemorySegment(memorySegmentDataset, size, dims))
                .withIndexParams(indexParams)
                .build();
            blackhole.consume(index);
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @BenchmarkMode(Mode.AverageTime)
    public void testDatasetFromHeap(Blackhole blackhole) throws Throwable {
        try (var dataset = Dataset.ofArray(arrayDataset)) {
            blackhole.consume(dataset);
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @BenchmarkMode(Mode.AverageTime)
    public void testDatasetFromMemorySegment(Blackhole blackhole) throws Throwable {
        try (var dataset = Dataset.ofMemorySegment(memorySegmentDataset, size, dims)) {
            blackhole.consume(dataset);
        }
    }
}
