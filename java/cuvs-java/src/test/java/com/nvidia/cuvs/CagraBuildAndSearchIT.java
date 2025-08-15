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
import static org.junit.Assert.*;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraMergeParams.MergeStrategy;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.LongToIntFunction;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class CagraBuildAndSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.trace("Random context initialized for test.");
  }

  private static void runConcurrently(int nThreads, Supplier<Runnable> runnableSupplier)
      throws ExecutionException, InterruptedException, TimeoutException {
    try (ExecutorService parallelExecutor = Executors.newFixedThreadPool(nThreads)) {
      var futures = new CompletableFuture[nThreads];
      for (int j = 0; j < nThreads; j++) {
        futures[j] = CompletableFuture.runAsync(runnableSupplier.get(), parallelExecutor);
      }

      CompletableFuture.allOf(futures)
          .exceptionally(
              t -> {
                log.error("Exception while executing runnable", t);
                fail("Exception while executing runnable: " + unwrap(t));
                return null;
              })
          .get(2000, TimeUnit.SECONDS);
    }
  }

  private static Throwable unwrap(Throwable t) {
    var root = t;
    while (root.getCause() != null) {
      root = root.getCause();
    }
    return root;
  }

  private static void runInAnotherThread(Runnable runnable)
      throws ExecutionException, InterruptedException, TimeoutException {
    try (ExecutorService singleExecutor = Executors.newSingleThreadExecutor()) {
      singleExecutor.submit(runnable).get(2000, TimeUnit.SECONDS);
    }
  }

  private static List<Map<Integer, Float>> getExpectedResults() {
    return Arrays.asList(
        Map.of(3, 0.038782578f, 2, 0.3590463f, 0, 0.83774555f),
        Map.of(0, 0.12472608f, 2, 0.21700792f, 1, 0.31918612f),
        Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f),
        Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.5986642f));
  }

  private static float[][] createSampleQueries() {
    return new float[][] {
      {0.48216683f, 0.0428398f},
      {0.5084142f, 0.6545497f},
      {0.51260436f, 0.2643005f},
      {0.05198065f, 0.5789965f}
    };
  }

  private static float[][] createSampleData() {
    return new float[][] {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f}
    };
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    int numTestsRuns = 5;
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      for (int j = 0; j < numTestsRuns; j++) {
        var index = indexOnce(CuVSMatrix.ofArray(dataset), resources);
        var indexPath = serializeOnce(index);
        var loadedIndex = deserializeOnce(indexPath, resources);
        queryAndCompare(
            index,
            loadedIndex,
            SearchResults.IDENTITY_MAPPING,
            queries,
            expectedResults,
            resources);
        cleanup(index, loadedIndex);
        Files.deleteIfExists(indexPath);
      }
    }
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlowInDifferentThreads() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    int numTestsRuns = 5;
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      for (int j = 0; j < numTestsRuns; j++) {
        runInAnotherThread(
            () -> {
              try {
                var index = indexOnce(CuVSMatrix.ofArray(dataset), resources);
                var indexPath = serializeOnce(index);
                var loadedIndex = deserializeOnce(indexPath, resources);
                queryAndCompare(
                    index,
                    loadedIndex,
                    SearchResults.IDENTITY_MAPPING,
                    queries,
                    expectedResults,
                    resources);
                cleanup(index, loadedIndex);
                Files.deleteIfExists(indexPath);
              } catch (Throwable e) {
                throw new RuntimeException(e);
              }
            });
      }
    }
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlowConcurrently() throws Throwable {
    final float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    int numTestsRuns = 10;

    runConcurrently(
        numTestsRuns,
        () ->
            () -> {
              try (CuVSResources resources = CheckedCuVSResources.create()) {
                var index = indexOnce(CuVSMatrix.ofArray(dataset), resources);
                var indexPath = serializeOnce(index);
                var loadedIndex = deserializeOnce(indexPath, resources);
                queryAndCompare(
                    index,
                    loadedIndex,
                    SearchResults.IDENTITY_MAPPING,
                    queries,
                    expectedResults,
                    resources);
                cleanup(index, loadedIndex);
                Files.deleteIfExists(indexPath);
              } catch (Throwable e) {
                throw new RuntimeException(e);
              }
            });
  }

  @Test
  public void testIndexing() throws Throwable {
    for (int i = 0; i < 100; ++i) {
      final float[][] dataset = createSampleData();
      int numTestsRuns = 10;
      runConcurrently(
          numTestsRuns,
          () ->
              () -> {
                try (CuVSResources resources = CheckedCuVSResources.create()) {
                  var index = indexOnce(CuVSMatrix.ofArray(dataset), resources);
                  index.close();
                } catch (Throwable e) {
                  throw new RuntimeException(e);
                }
              });
    }
  }

  @Test
  public void testSerialization() throws Throwable {
    for (int i = 0; i < 100; ++i) {
      final float[][] dataset = createSampleData();
      int numTestsRuns = 10;
      runConcurrently(
          numTestsRuns,
          () ->
              () -> {
                try (CuVSResources resources = CheckedCuVSResources.create();
                    var index = indexOnce(CuVSMatrix.ofArray(dataset), resources)) {
                  var indexPath = serializeOnce(index);
                  Files.deleteIfExists(indexPath);
                } catch (Throwable e) {
                  throw new RuntimeException(e);
                }
              });
    }
  }

  @Test
  public void testDeserialization() throws Throwable {
    var indexPath = createSerializedIndex(CuVSMatrix.ofArray(createSampleData()));
    for (int i = 0; i < 100; ++i) {
      int numTestsRuns = 10;
      runConcurrently(
          numTestsRuns,
          () ->
              () -> {
                try (CuVSResources resources = CheckedCuVSResources.create()) {
                  deserializeOnce(indexPath, resources).close();
                } catch (Throwable e) {
                  throw new RuntimeException(e);
                }
              });
    }
    Files.deleteIfExists(indexPath);
  }

  private Path createSerializedIndex(CuVSMatrix dataset) throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        var index = indexOnce(dataset, resources)) {
      return serializeOnce(index);
    }
  }

  @Test
  public void testReconstructIndexFromGraph() throws Throwable {
    try (var dataset = CuVSMatrix.ofArray(createSampleData())) {
      var queries = createSampleQueries();
      List<Map<Integer, Float>> expectedResults = getExpectedResults();

      try (CuVSResources resources = CuVSResources.create();
          var index = indexOnce(dataset, resources)) {
        var graph = index.getGraph();

        try (var reconstructedIndex =
            CagraIndex.newBuilder(resources)
                .from(graph)
                .withDataset(dataset)
                .withIndexParams(
                    new CagraIndexParams.Builder().withMetric(CuvsDistanceType.L2Expanded).build())
                .build()) {
          queryAndCompare(
              index,
              reconstructedIndex,
              SearchResults.IDENTITY_MAPPING,
              queries,
              expectedResults,
              resources);

          var originalIndexPath = serializeOnce(index);
          var reconstructedIndexPath = serializeOnce(reconstructedIndex);

          var originalBytes = Files.readAllBytes(originalIndexPath);
          var reconstructedBytes = Files.readAllBytes(reconstructedIndexPath);

          assertArrayEquals(originalBytes, reconstructedBytes);

          Files.deleteIfExists(originalIndexPath);
          Files.deleteIfExists(reconstructedIndexPath);
        }
      }
    }
  }

  @Test
  public void testIndexingAndSearchingFlowWithCustomMappingFunction() throws Throwable {
    var dataset = CuVSMatrix.ofArray(createSampleData());
    float[][] queries = createSampleQueries();
    var expectedResults =
        List.of(
            Map.of(0, 0.038782578f, 3, 0.3590463f, 1, 0.83774555f),
            Map.of(1, 0.12472608f, 3, 0.21700792f, 2, 0.31918612f),
            Map.of(0, 0.047766715f, 3, 0.20332818f, 1, 0.48305473f),
            Map.of(2, 0.15224178f, 1, 0.59063464f, 0, 0.5986642f));

    LongToIntFunction rotate = l -> (int) ((l + 1) % dataset.size());
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var index = indexOnce(dataset, resources);
      var indexPath = serializeOnce(index);
      var loadedIndex = deserializeOnce(indexPath, resources);
      queryAndCompare(index, loadedIndex, rotate, queries, expectedResults, resources);
      cleanup(index, loadedIndex);
      Files.deleteIfExists(indexPath);
    }
  }

  @Test
  public void testIndexingAndSearchingFlowWithCustomMappingList() throws Throwable {
    var dataset = CuVSMatrix.ofArray(createSampleData());
    float[][] queries = createSampleQueries();
    var mappings = List.of(4, 3, 2, 1);
    var expectedResults =
        List.of(
            Map.of(1, 0.038782578f, 2, 0.3590463f, 4, 0.83774555f),
            Map.of(4, 0.12472608f, 2, 0.21700792f, 3, 0.31918612f),
            Map.of(1, 0.047766715f, 2, 0.20332818f, 4, 0.48305473f),
            Map.of(3, 0.15224178f, 4, 0.59063464f, 1, 0.5986642f));

    LongToIntFunction rotate = SearchResults.mappingsFromList(mappings);
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var index = indexOnce(dataset, resources);
      var indexPath = serializeOnce(index);
      var loadedIndex = deserializeOnce(indexPath, resources);
      queryAndCompare(index, loadedIndex, rotate, queries, expectedResults, resources);
      cleanup(index, loadedIndex);
      Files.deleteIfExists(indexPath);
    }
  }

  /**
   * A test that checks the pre-filtering feature.
   */
  @Test
  public void testPrefilteringReducesResults() throws Throwable {

    // Sample data and query
    float[][] dataset = createSampleData();
    float[][] queries = {{0.48216683f, 0.0428398f}};

    // Expected search results
    List<Map<Integer, Float>> expectedResults = List.of(Map.of(3, 0.038782578f, 2, 0.3590463f));

    // Expected filtered search results
    List<Map<Integer, Float>> expectedFilteredResults =
        List.of(Map.of(2, 0.3590463f, 0, 0.83774555f));

    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(2)
            .withIntermediateGraphDegree(4)
            .withNumWriterThreads(2)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build();

      // No prefilter (all points allowed)
      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
      CagraQuery fullQuery =
          new CagraQuery.Builder(resources)
              .withTopK(2)
              .withSearchParams(searchParams)
              .withQueryVectors(queries)
              .build();

      SearchResults fullSearchResults = index.search(fullQuery);
      List<Map<Integer, Float>> fullResults = fullSearchResults.getResults();
      log.debug("Full results: {}", fullResults);

      // Apply prefilter: only allow ids 0 and 2 (bitset: 1100)
      BitSet prefilter = new BitSet(4);
      prefilter.set(0);
      prefilter.set(2);

      CagraQuery filteredQuery =
          new CagraQuery.Builder(resources)
              .withTopK(2)
              .withSearchParams(searchParams)
              .withQueryVectors(queries)
              .withPrefilter(prefilter, 4)
              .build();

      SearchResults filteredSearchResults = index.search(filteredQuery);
      List<Map<Integer, Float>> filteredResults = filteredSearchResults.getResults();
      log.debug("Filtered results: {}", filteredResults);

      assertEquals(expectedResults, fullResults);
      assertEquals(expectedFilteredResults, filteredResults);
    }
  }

  private CagraIndex indexOnce(CuVSMatrix dataset, CuVSResources resources) throws Throwable {
    // Configure index parameters
    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(1)
            .withIntermediateGraphDegree(2)
            .withNumWriterThreads(32)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    // Create the index with the dataset
    return CagraIndex.newBuilder(resources)
        .withDataset(dataset)
        .withIndexParams(indexParams)
        .build();
  }

  private Path serializeOnce(CagraIndex index) throws Throwable {
    // Saving the index on to the disk.
    var indexFilePath = Path.of(UUID.randomUUID() + ".cag");
    try (var outputStream = Files.newOutputStream(indexFilePath)) {
      index.serialize(outputStream);
    }
    return indexFilePath;
  }

  private CagraIndex deserializeOnce(Path indexFilePath, CuVSResources resources) throws Throwable {
    // Loading a CAGRA index from disk.
    try (var inputStream = Files.newInputStream(indexFilePath)) {
      return CagraIndex.newBuilder(resources).from(inputStream).build();
    }
  }

  private void queryAndCompare(
      CagraIndex index1,
      CagraIndex index2,
      LongToIntFunction mapping,
      float[][] queries,
      List<Map<Integer, Float>> expectedResults,
      CuVSResources resources)
      throws Throwable {
    // Configure search parameters
    CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

    // Create a query object with the query vectors
    CagraQuery cuvsQuery =
        new CagraQuery.Builder(resources)
            .withTopK(3)
            .withSearchParams(searchParams)
            .withQueryVectors(queries)
            .withMapping(mapping)
            .build();

    // Perform the search
    SearchResults results = index1.search(cuvsQuery);

    // Check results
    log.debug(results.getResults().toString());
    checkResults(expectedResults, results.getResults());

    // Search from the second index
    results = index2.search(cuvsQuery);

    // Check results
    log.debug(results.getResults().toString());
    checkResults(expectedResults, results.getResults());
  }

  private void cleanup(CagraIndex index, CagraIndex loadedIndex) throws Throwable {
    // Cleanup
    index.close();
    loadedIndex.close();
  }

  @Test
  public void testNativeDatasetEquivalent() throws Throwable {
    float[][] sampleData = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    ValueLayout.OfFloat C_FLOAT =
        (ValueLayout.OfFloat) Linker.nativeLinker().canonicalLayouts().get("float");

    int rows = sampleData.length;
    int cols = sampleData[0].length;
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout((long) rows * cols, C_FLOAT);

    try (Arena arena = Arena.ofShared()) {
      MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
      for (int r = 0; r < rows; r++) {
        MemorySegment.copy(
            sampleData[r], 0, dataMemorySegment, C_FLOAT, (r * cols * C_FLOAT.byteSize()), cols);
      }

      try (var resources = CuVSResources.create();
          var javaDataset = CuVSMatrix.ofArray(sampleData);
          var nativeDataset =
              DatasetHelper.fromMemorySegment(
                  dataMemorySegment, rows, cols, CuVSMatrix.DataType.FLOAT)) {

        // Indexing with an on-heap and native datasets produce the same results
        var javaIndex = indexOnce(javaDataset, resources);
        var nativeIndex = indexOnce(nativeDataset, resources);
        queryAndCompare(
            javaIndex,
            nativeIndex,
            SearchResults.IDENTITY_MAPPING,
            queries,
            expectedResults,
            resources);
        cleanup(javaIndex, nativeIndex);
      }
    }
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
      {1.0f, 1.0f}, // Should be closest to vector1[1] -> index 1
      {10.5f, 10.5f}, // Should be closest to vector2[0] -> index 2
      {0.0f, 0.0f} // Should be closest to vector1[0] -> index 0
    };

    // Expected search results for each query (nearest neighbor and its distance)
    List<Map<Integer, Float>> expectedResults =
        Arrays.asList(
            Map.of(1, 0.0f, 0, 2.0f, 2, 162.0f),
            Map.of(2, 0.5f, 3, 0.5f, 1, 180.5f),
            Map.of(0, 0.0f, 1, 2.0f, 2, 200.0f));

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      log.trace("Building first index...");
      CagraIndex index1 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector1)
              .withIndexParams(indexParams)
              .build();

      log.trace("Building second index...");
      CagraIndex index2 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector2)
              .withIndexParams(indexParams)
              .build();

      log.trace("Merging indexes...");
      CagraIndex mergedIndex = CagraIndex.merge(new CagraIndex[] {index1, index2});
      log.trace("Merge completed successfully");

      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

      CagraQuery query =
          new CagraQuery.Builder(resources)
              .withTopK(3)
              .withSearchParams(searchParams)
              .withQueryVectors(queries)
              .withMapping(SearchResults.IDENTITY_MAPPING)
              .build();

      log.trace("Searching merged index...");
      SearchResults results = mergedIndex.search(query);
      log.debug("Search results: " + results.getResults().toString());

      assertEquals(expectedResults, results.getResults());

      // --- Serialization/deserialization check ---
      String indexFileName = UUID.randomUUID() + ".cag";
      mergedIndex.serialize(new FileOutputStream(indexFileName));

      File indexFile = new File(indexFileName);
      InputStream inputStream = new FileInputStream(indexFile);
      CagraIndex loadedMergedIndex = CagraIndex.newBuilder(resources).from(inputStream).build();

      SearchResults resultsFromLoaded = loadedMergedIndex.search(query);
      assertEquals(expectedResults, resultsFromLoaded.getResults());

      if (indexFile.exists()) {
        indexFile.delete();
      }
      index1.close();
      index2.close();
      mergedIndex.close();
      loadedMergedIndex.close();
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

    List<Map<Integer, Float>> expectedResults =
        Arrays.asList(
            Map.of(1, 0.0f, 0, 2.0f, 2, 162.0f),
            Map.of(2, 0.5f, 3, 0.5f, 1, 180.5f),
            Map.of(0, 0.0f, 1, 2.0f, 2, 200.0f));

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      log.trace("Building first index...");
      CagraIndex index1 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector1)
              .withIndexParams(indexParams)
              .build();

      log.trace("Building second index...");
      CagraIndex index2 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector2)
              .withIndexParams(indexParams)
              .build();

      CagraIndexParams outputIndexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(2)
              .withIntermediateGraphDegree(4)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraMergeParams physicalMergeParams =
          new CagraMergeParams.Builder()
              .withOutputIndexParams(outputIndexParams)
              .withStrategy(MergeStrategy.PHYSICAL)
              .build();

      log.trace("Merging indexes with PHYSICAL strategy...");
      CagraIndex physicalMergedIndex =
          CagraIndex.merge(new CagraIndex[] {index1, index2}, physicalMergeParams);
      log.trace("Physical merge completed successfully");

      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

      CagraQuery query =
          new CagraQuery.Builder(resources)
              .withTopK(3)
              .withSearchParams(searchParams)
              .withQueryVectors(queries)
              .withMapping(SearchResults.IDENTITY_MAPPING)
              .build();

      log.trace("Searching physically merged index...");
      SearchResults physicalResults = physicalMergedIndex.search(query);
      assertNotNull("Physical merge search results should not be null", physicalResults);
      assertEquals(
          "Physical merge search results should match expected",
          expectedResults,
          physicalResults.getResults());

      // --- Serialization/deserialization check for both merged indexes ---
      String physicalIndexFileName = UUID.randomUUID().toString() + ".cag";
      physicalMergedIndex.serialize(new FileOutputStream(physicalIndexFileName));

      File physicalIndexFile = new File(physicalIndexFileName);
      InputStream physicalInputStream = new FileInputStream(physicalIndexFile);
      CagraIndex loadedPhysicalIndex =
          CagraIndex.newBuilder(resources).from(physicalInputStream).build();

      SearchResults resultsFromLoadedPhysical = loadedPhysicalIndex.search(query);
      assertEquals(
          "Loaded physical index search results should match expected",
          expectedResults,
          resultsFromLoadedPhysical.getResults());

      if (physicalIndexFile.exists()) {
        physicalIndexFile.delete();
      }
      index1.close();
      index2.close();
      physicalMergedIndex.close();
      loadedPhysicalIndex.close();
    }
  }
}
