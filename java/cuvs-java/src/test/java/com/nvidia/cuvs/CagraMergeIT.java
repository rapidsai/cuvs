package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;

@RunWith(RandomizedRunner.class)
public class CagraMergeIT extends CuVSTestCase {

    private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    @Before
    public void setup() {
        assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
        initializeRandom();
        log.info("Random context initialized for test.");
    }

    @Test
    public void testMergingIndexes() throws Throwable {
        float[][] dataset1 = {
            {0.0f, 0.0f}, 
            {1.0f, 1.0f}
        };

        float[][] dataset2 = {
            {10.0f, 10.0f}, 
            {11.0f, 11.0f}
        };

        float[][] queries = {
            {1.0f, 1.0f},    // Should be closest to dataset1[1] -> index 1
            {10.5f, 10.5f},  // Should be closest to dataset2[0] -> index 2
            {0.0f, 0.0f}     // Should be closest to dataset1[0] -> index 0
        };

        // Expected search results for each query (nearest neighbor and its distance)
        List<Map<Integer, Float>> expectedResults = Arrays.asList(
            Map.of(1, 0.0f, 0, 2.0f, 2, 162.0f),
            Map.of(2, 0.5f, 3, 0.5f, 1, 180.5f),
            Map.of(0, 0.0f, 1, 2.0f, 2, 200.0f)
        );

        try (CuVSResources resources = CuVSResources.create()) {
            CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(1)
                .withIntermediateGraphDegree(2)
                .withNumWriterThreads(4)
                .withMetric(CuvsDistanceType.L2Expanded)
                .build();

            log.info("Building first index...");
            CagraIndex index1 = CagraIndex.newBuilder(resources)
                .withDataset(dataset1)
                .withIndexParams(indexParams)
                .build();

            log.info("Building second index...");
            CagraIndex index2 = CagraIndex.newBuilder(resources)
                .withDataset(dataset2)
                .withIndexParams(indexParams)
                .build();

            log.info("Merging indexes...");
            CagraIndex mergedIndex = CagraIndex.merge(new CagraIndex[]{index1, index2});
            log.info("Merge completed successfully");

            CagraSearchParams searchParams = new CagraSearchParams.Builder(resources)
                .build();

            List<Integer> mergedMap = Arrays.asList(0, 1, 2, 3);
            CagraQuery query = new CagraQuery.Builder()
                .withTopK(3)
                .withSearchParams(searchParams)
                .withQueryVectors(queries)
                .withMapping(mergedMap)
                .build();

            // Search using the merged index
                log.info("Searching merged index...");
                SearchResults results = mergedIndex.search(query);
                log.info("Search results: " + results.getResults().toString());

                // Direct, strict check of the results
                assertEquals(expectedResults, results.getResults());

                // --- Serialization/deserialization check ---
                String indexFileName = java.util.UUID.randomUUID().toString() + ".cag";
                mergedIndex.serialize(new java.io.FileOutputStream(indexFileName));

                java.io.File indexFile = new java.io.File(indexFileName);
                java.io.InputStream inputStream = new java.io.FileInputStream(indexFile);
                CagraIndex loadedMergedIndex = CagraIndex.newBuilder(resources)
                    .from(inputStream)
                    .build();

                SearchResults resultsFromLoaded = loadedMergedIndex.search(query);
                assertEquals(expectedResults, resultsFromLoaded.getResults());

                // Cleanup
                if (indexFile.exists()) {
                    indexFile.delete();
                }
                index1.destroyIndex();
                index2.destroyIndex();
                mergedIndex.destroyIndex();
                loadedMergedIndex.destroyIndex();

        }
    }
}