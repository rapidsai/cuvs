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

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.BitSet;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraMergeParams.MergeStrategy;

@RunWith(RandomizedRunner.class)
public class CagraBuildAndSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.info("Random context initialized for test.");
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   *
   * @throws Throwable
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {

    // Sample data and query
    float[][] dataset = {
        { 0.74021935f, 0.9209938f },
        { 0.03902049f, 0.9689629f },
        { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f }
    };
    List<Integer> map = List.of(0, 1, 2, 3);
    float[][] queries = {
        { 0.48216683f, 0.0428398f },
        { 0.5084142f, 0.6545497f },
        { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f }
    };

    // Expected search results
    List<Map<Integer, Float>> expectedResults = Arrays.asList(
        Map.of(3, 0.038782578f, 2, 0.3590463f, 0, 0.83774555f),
        Map.of(0, 0.12472608f, 2, 0.21700792f, 1, 0.31918612f),
        Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f),
        Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.5986642f));

    int numTestsRuns = 10;

    try (CuVSResources resources = CuVSResources.create()) {
      // sometimes run this test using different threads?
      boolean runTestInDifferentThreads = random.nextBoolean();
      // if running in different threads, run concurrently or one after the other?
      boolean runConcurrently = runTestInDifferentThreads ? random.nextBoolean(): false;

      log.info("Running in different threads? " + runTestInDifferentThreads);
      log.info("Running concurrently? " + runConcurrently);

      ExecutorService parallelExecutor = runConcurrently ? Executors.newFixedThreadPool(numTestsRuns): null;

      for (int j = 0; j < numTestsRuns; j++) {
        Runnable testLogic = indexAndQueryOnce(dataset, map, queries, expectedResults, resources);
        if (runTestInDifferentThreads) {
          if (runConcurrently) {
            parallelExecutor.submit(testLogic);
          } else {
            ExecutorService singleExecutor = Executors.newSingleThreadExecutor();
            singleExecutor.submit(testLogic);
            singleExecutor.shutdown();
            singleExecutor.awaitTermination(2000, TimeUnit.SECONDS);
          }
        } else {
          // run the test logic in the main thread
          testLogic.run();
        }
      }
      if (parallelExecutor != null) {
        parallelExecutor.shutdown();
        parallelExecutor.awaitTermination(2000, TimeUnit.SECONDS);
      }

    }
  }

  /**
   * A test that checks the pre-filtering feature.
   *
   * @throws Throwable
   */
  @Test
  public void testPrefilteringReducesResults() throws Throwable {

    // Sample data and query
    float[][] dataset = {
        { 0.74021935f, 0.9209938f },
        { 0.03902049f, 0.9689629f },
        { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f }
       };
    float[][] queries = {
        { 0.48216683f, 0.0428398f }
       };

    // Expected search results
    List<Map<Integer, Float>> expectedResults = Arrays.asList(
        Map.of(3, 0.038782578f, 2, 0.3590463f));

    // Expected filtered search results
    List<Map<Integer, Float>> expectedFilteredResults = Arrays.asList(
        Map.of(2, 0.3590463f, 0, 0.83774555f));

    CagraIndexParams indexParams = new CagraIndexParams.Builder()
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .withGraphDegree(2)
        .withIntermediateGraphDegree(4)
        .withNumWriterThreads(2)
        .withMetric(CuvsDistanceType.L2Expanded)
        .build();

    try (CuVSResources resources = CuVSResources.create()) {
      CagraIndex index = CagraIndex.newBuilder(resources)
          .withDataset(dataset)
          .withIndexParams(indexParams)
          .build();

      // No prefilter (all points allowed)
      CagraSearchParams searchParams = new CagraSearchParams.Builder(resources)
          .build();
      CagraQuery fullQuery = new CagraQuery.Builder()
          .withTopK(2)
          .withSearchParams(searchParams)
          .withQueryVectors(queries)
          .build();

      SearchResults fullSearchResults = index.search(fullQuery);
      List<Map<Integer, Float>> fullResults = fullSearchResults.getResults();
      log.info("Full results: {}", fullResults);

      // Apply prefilter: only allow ids 0 and 2 (bitset: 1100)
      BitSet prefilter = new BitSet(4);
      prefilter.set(0);
      prefilter.set(2);

      CagraQuery filteredQuery = new CagraQuery.Builder()
          .withTopK(2)
          .withSearchParams(searchParams)
          .withQueryVectors(queries)
          .withPrefilter(prefilter, 4)
          .build();

      SearchResults filteredSearchResults = index.search(filteredQuery);
      List<Map<Integer, Float>> filteredResults = filteredSearchResults.getResults();
      log.info("Filtered results: {}", filteredResults);

      assertEquals(expectedResults, fullResults);
      assertEquals(expectedFilteredResults, filteredResults);
    }
  }

  private Runnable indexAndQueryOnce(float[][] dataset, List<Integer> map, float[][] queries,
      List<Map<Integer, Float>> expectedResults, CuVSResources resources) throws Throwable, FileNotFoundException {

    Runnable thread = new Runnable() {

      @Override
      public void run() {
        try {

          // Configure index parameters
          CagraIndexParams indexParams = new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(32)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

          // Create the index with the dataset
          CagraIndex index = CagraIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build();

          // Saving the index on to the disk.
          String indexFileName = UUID.randomUUID().toString() + ".cag";
          index.serialize(new FileOutputStream(indexFileName));

          // Loading a CAGRA index from disk.
          File indexFile = new File(indexFileName);
          InputStream inputStream = new FileInputStream(indexFile);
          CagraIndex loadedIndex = CagraIndex.newBuilder(resources)
              .from(inputStream)
              .build();

          // Configure search parameters
          CagraSearchParams searchParams = new CagraSearchParams.Builder(resources)
              .build();

          // Create a query object with the query vectors
          CagraQuery cuvsQuery = new CagraQuery.Builder()
              .withTopK(3)
              .withSearchParams(searchParams)
              .withQueryVectors(queries)
              .withMapping(map)
              .build();

          // Perform the search
          SearchResults results = index.search(cuvsQuery);

          // Check results
          log.info(results.getResults().toString());
          assertEquals(expectedResults, results.getResults());

          // Search from deserialized index
          results = loadedIndex.search(cuvsQuery);

          // Check results
          log.info(results.getResults().toString());
          assertEquals(expectedResults, results.getResults());

          // Cleanup
          if (indexFile.exists()) {
            indexFile.delete();
          }
          index.destroyIndex();
        } catch (Throwable ex) {
          throw new RuntimeException("Exception during indexing/querying", ex);
        }
      }
    };
    return thread;
  }

  @Test
    public void testMergingIndexes() throws Throwable {
        float[][] vector1 = {
            {0.0f, 0.0f},
            {1.0f, 1.0f}
        };

        float[][] vector2 = {
            {10.0f, 10.0f},
            {11.0f, 11.0f}
        };

        float[][] queries = {
            {1.0f, 1.0f},    // Should be closest to vector1[1] -> index 1
            {10.5f, 10.5f},  // Should be closest to vector2[0] -> index 2
            {0.0f, 0.0f}     // Should be closest to vector1[0] -> index 0
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
                .withDataset(vector1)
                .withIndexParams(indexParams)
                .build();

            log.info("Building second index...");
            CagraIndex index2 = CagraIndex.newBuilder(resources)
                .withDataset(vector2)
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

                log.info("Searching merged index...");
                SearchResults results = mergedIndex.search(query);
                log.info("Search results: " + results.getResults().toString());

                assertEquals(expectedResults, results.getResults());

                // --- Serialization/deserialization check ---
                String indexFileName = UUID.randomUUID().toString() + ".cag";
                mergedIndex.serialize(new FileOutputStream(indexFileName));

                File indexFile = new File(indexFileName);
                InputStream inputStream = new FileInputStream(indexFile);
                CagraIndex loadedMergedIndex = CagraIndex.newBuilder(resources)
                    .from(inputStream)
                    .build();

                SearchResults resultsFromLoaded = loadedMergedIndex.search(query);
                assertEquals(expectedResults, resultsFromLoaded.getResults());

                if (indexFile.exists()) {
                    indexFile.delete();
                }
                index1.destroyIndex();
                index2.destroyIndex();
                mergedIndex.destroyIndex();
                loadedMergedIndex.destroyIndex();

        }
    }

    // Commented out test for Logical merge strategy as it is not yet implemented in C yet
    @Test
    public void testMergeStrategies() throws Throwable {
        float[][] vector1 = {
            {0.0f, 0.0f},
            {1.0f, 1.0f}
        };

        float[][] vector2 = {
            {10.0f, 10.0f},
            {11.0f, 11.0f}
        };

        float[][] queries = {
            {1.0f, 1.0f},
            {10.5f, 10.5f},
            {0.0f, 0.0f}
        };

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
                .withDataset(vector1)
                .withIndexParams(indexParams)
                .build();

            log.info("Building second index...");
            CagraIndex index2 = CagraIndex.newBuilder(resources)
                .withDataset(vector2)
                .withIndexParams(indexParams)
                .build();

            CagraIndexParams outputIndexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(2)
                .withIntermediateGraphDegree(4)
                .withNumWriterThreads(4)
                .withMetric(CuvsDistanceType.L2Expanded)
                .build();

            CagraMergeParams physicalMergeParams = new CagraMergeParams.Builder()
                .withOutputIndexParams(outputIndexParams)
                .withStrategy(MergeStrategy.PHYSICAL)
                .build();

            log.info("Merging indexes with PHYSICAL strategy...");
            CagraIndex physicalMergedIndex = CagraIndex.merge(new CagraIndex[]{index1, index2}, physicalMergeParams);
            log.info("Physical merge completed successfully");

            CagraSearchParams searchParams = new CagraSearchParams.Builder(resources)
                .build();

            List<Integer> mergedMap = Arrays.asList(0, 1, 2, 3);
            CagraQuery query = new CagraQuery.Builder()
                .withTopK(3)
                .withSearchParams(searchParams)
                .withQueryVectors(queries)
                .withMapping(mergedMap)
                .build();

            log.info("Searching physically merged index...");
            SearchResults physicalResults = physicalMergedIndex.search(query);
            assertNotNull("Physical merge search results should not be null", physicalResults);
            assertEquals("Physical merge search results should match expected", expectedResults, physicalResults.getResults());

            // --- Serialization/deserialization check for both merged indexes ---
            String physicalIndexFileName = UUID.randomUUID().toString() + ".cag";
            physicalMergedIndex.serialize(new FileOutputStream(physicalIndexFileName));

            File physicalIndexFile = new File(physicalIndexFileName);
            InputStream physicalInputStream = new FileInputStream(physicalIndexFile);
            CagraIndex loadedPhysicalIndex = CagraIndex.newBuilder(resources)
                .from(physicalInputStream)
                .build();

            SearchResults resultsFromLoadedPhysical = loadedPhysicalIndex.search(query);
            assertEquals("Loaded physical index search results should match expected",
                expectedResults, resultsFromLoadedPhysical.getResults());

            if (physicalIndexFile.exists()) {
                physicalIndexFile.delete();
            }
            index1.destroyIndex();
            index2.destroyIndex();
            physicalMergedIndex.destroyIndex();
            loadedPhysicalIndex.destroyIndex();
        }
    }

}
