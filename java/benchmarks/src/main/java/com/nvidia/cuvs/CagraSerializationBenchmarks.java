package com.nvidia.cuvs;

import org.openjdk.jmh.annotations.*;

import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static com.nvidia.cuvs.Utils.createSampleData;

@Fork(value = 1, warmups = 0)
@State(Scope.Benchmark)
public class CagraSerializationBenchmarks {

    @Param({ "1024" })
    private int dims;

    @Param({ "100" })
    private int size;

    private Arena arena;

    private CuVSResources resources;

    private CagraIndex index;

    @Setup
    public void initialize() throws Throwable {
        resources = CuVSResources.create();
        arena = Arena.ofShared();
        var arrayDataset = createSampleData(size, dims);

        // Configure index parameters
        CagraIndexParams indexParams = new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(1)
            .withIntermediateGraphDegree(2)
            .withNumWriterThreads(32)
            .withMetric(CagraIndexParams.CuvsDistanceType.L2Expanded)
            .build();

        // Create the index with the dataset
        index = CagraIndex.newBuilder(resources)
            .withDataset(arrayDataset)
            .withIndexParams(indexParams)
            .build();
    }

    @TearDown
    public void cleanUp() throws Throwable {
        if (resources != null) {
            resources.close();
        }
        if (arena != null) {
            arena.close();
        }
        if (index != null) {
            index.destroyIndex();
        }
    }

    @Benchmark
    public void testSerializeToFile() throws Throwable {
        var indexFilePath = Path.of(UUID.randomUUID() + ".cag");
        try (var outputStream = Files.newOutputStream(indexFilePath)) {
            index.serialize(outputStream);
        }
        Files.deleteIfExists(indexFilePath);
    }

    @Benchmark
    public void testSerializeToMemory() throws Throwable {
        try (var arena = Arena.ofConfined()) {
            var buffer = arena.allocate(1024 * 1024);
            index.serialize((Object) buffer);
            // TODO: reinterpret should happen in serialize, with data from the C call

        }
    }
}
