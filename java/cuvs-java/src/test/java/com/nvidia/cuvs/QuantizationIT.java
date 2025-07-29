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
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class QuantizationIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.info("Random context initialized for quantization test.");
  }

  private static float[][] createSampleData() {
    return new float[][] {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f}
    };
  }

  private static float[][] createSampleQueries() {
    return new float[][] {
      {0.48216683f, 0.0428398f},
      {0.5084142f, 0.6545497f}
    };
  }

  /**
   * Test scalar 8-bit quantization workflow: indexing with quantized dataset and searching with quantized queries
   */
  @Test
  public void testScalar8BitQuantizationFlow() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {

      // 1. Create training dataset
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      assertEquals("Training dataset should be 32-bit", 32, trainingDataset.precision());

      // 2. Create scalar quantizer
      CuVSQuantizer quantizer = new Scalar8BitQuantizer(resources, trainingDataset);
      assertEquals("Quantizer should have 8-bit precision", 8, quantizer.precision());
      log.info("Created Scalar8BitQuantizer with {}-bit precision", quantizer.precision());

      // 3. Build index with quantized dataset using withQuantizer()
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(32)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraIndex quantizedIndex =
          CagraIndex.newBuilder(resources)
              .withDataset(trainingDataset) // 32-bit input
              .withQuantizer(quantizer) // automatically quantizes to 8-bit
              .withIndexParams(indexParams)
              .build();

      log.info("✓ Index built with quantized dataset");

      // 4. Test search with quantized queries
      CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();

      CagraQuery quantizedQuery =
          CagraQuery.newBuilder()
              .withQueryVectors(queries) // 32-bit input
              .withQuantizer(quantizer) // automatically quantizes to 8-bit
              .withSearchParams(searchParams)
              .withTopK(3)
              .build();

      // Verify query was properly quantized
      assertTrue("Query should have quantized vectors", quantizedQuery.hasQuantizedQueries());
      assertEquals(
          "Quantized query should have 8-bit precision", 8, quantizedQuery.getQueryPrecision());

      SearchResults quantizedResults = quantizedIndex.search(quantizedQuery);
      assertNotNull("Quantized query results should not be null", quantizedResults);

      List<Map<Integer, Float>> results = quantizedResults.getResults();
      assertEquals("Should return results for all queries", queries.length, results.size());
      assertTrue("Each query should return some results", results.get(0).size() > 0);

      log.info(
          "✓ Quantized search completed: {} queries, {} results each",
          results.size(),
          results.get(0).size());

      // 5. Test serialization/deserialization works with quantized index
      Path indexPath = serializeIndex(quantizedIndex);
      CagraIndex loadedIndex = deserializeIndex(indexPath, resources);

      SearchResults loadedResults = loadedIndex.search(quantizedQuery);
      assertEquals(
          "Loaded index should return same number of queries",
          results.size(),
          loadedResults.getResults().size());

      log.info("✓ Quantized index serialization/deserialization works");

      // 6. Test inverse transform
      CuVSMatrix quantizedQueries = quantizedQuery.getQuantizedQueries();
      CuVSMatrix recoveredQueries = quantizer.inverseTransform(quantizedQueries);
      assertNotNull("Inverse transform should work", recoveredQueries);
      assertEquals("Should recover all queries", queries.length, recoveredQueries.size());

      // Cleanup
      quantizedIndex.destroyIndex();
      loadedIndex.destroyIndex();
      quantizer.close();
      trainingDataset.close();
      quantizedQueries.close();
      Files.deleteIfExists(indexPath);

      log.info("✓ Scalar 8-bit quantization test completed successfully");
    }
  }

  /**
   * Test binary quantization workflow: indexing and searching with 1-bit quantized data
   */
  @Test
  public void testBinaryQuantizationFlow() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {

      // 1. Create training dataset
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      assertEquals("Training dataset should be 32-bit", 32, trainingDataset.precision());

      // 2. Create binary quantizer
      CuVSQuantizer binaryQuantizer = new BinaryQuantizer(resources);
      assertEquals("Binary quantizer should have 8-bit precision", 8, binaryQuantizer.precision());
      log.info("Created BinaryQuantizer with {}-bit precision", binaryQuantizer.precision());

      // 3. Build index with binary quantized dataset
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(32)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraIndex binaryIndex =
          CagraIndex.newBuilder(resources)
              .withDataset(trainingDataset) // 32-bit input
              .withQuantizer(binaryQuantizer) // automatically quantizes to 1-bit
              .withIndexParams(indexParams)
              .build();

      log.info("✓ Index built with binary quantized dataset");

      // 4. Search with binary quantized queries
      CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();

      CagraQuery binaryQuery =
          CagraQuery.newBuilder()
              .withQueryVectors(queries) // 32-bit input
              .withQuantizer(binaryQuantizer) // automatically quantizes to 1-bit
              .withSearchParams(searchParams)
              .withTopK(3)
              .build();

      // Verify query was properly quantized
      assertTrue("Query should have quantized vectors", binaryQuery.hasQuantizedQueries());
      assertEquals("Binary query should have 8-bit precision", 8, binaryQuery.getQueryPrecision());

      SearchResults binaryResults = binaryIndex.search(binaryQuery);
      assertNotNull("Binary query results should not be null", binaryResults);

      List<Map<Integer, Float>> results = binaryResults.getResults();
      assertEquals("Should return results for all queries", queries.length, results.size());
      assertTrue("Each query should return some results", results.get(0).size() > 0);

      log.info(
          "✓ Binary quantized search completed: {} queries, {} results each",
          results.size(),
          results.get(0).size());

      // 5. Verify binary quantizer doesn't support inverse transform
      CuVSMatrix binaryQueries = binaryQuery.getQuantizedQueries();
      try {
        binaryQuantizer.inverseTransform(binaryQueries);
        assertTrue("Binary quantizer should not support inverse transform", false);
      } catch (UnsupportedOperationException e) {
        log.info("✓ Binary quantizer correctly doesn't support inverse transform");
      }

      // Cleanup
      binaryIndex.destroyIndex();
      binaryQuantizer.close();
      trainingDataset.close();
      binaryQueries.close();

      log.info("✓ Binary quantization test completed successfully");
    }
  }

  private Path serializeIndex(CagraIndex index) throws Throwable {
    Path indexFilePath = Path.of(UUID.randomUUID() + ".cag");
    try (var outputStream = Files.newOutputStream(indexFilePath)) {
      index.serialize(outputStream);
    }
    return indexFilePath;
  }

  private CagraIndex deserializeIndex(Path indexFilePath, CuVSResources resources)
      throws Throwable {
    try (var inputStream = Files.newInputStream(indexFilePath)) {
      return CagraIndex.newBuilder(resources).from(inputStream).build();
    }
  }

  /**
   * Test that verifies quantized search results have reasonable recall compared to baseline
   */
  @Test
  public void testQuantizedSearchQuality() throws Throwable {
    float[][] dataset = createLargerSampleData(); // Use larger dataset for meaningful recall
    float[][] queries = createSampleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // 1. Build baseline (non-quantized) index and get reference results
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(2)
              .withIntermediateGraphDegree(4)
              .withNumWriterThreads(32)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);

      // Build baseline index
      CagraIndex baselineIndex =
          CagraIndex.newBuilder(resources)
              .withDataset(trainingDataset)
              .withIndexParams(indexParams)
              .build();

      // Get baseline results
      CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();
      CagraQuery baselineQuery =
          CagraQuery.newBuilder()
              .withQueryVectors(queries)
              .withSearchParams(searchParams)
              .withTopK(10) // Use higher K for recall calculation
              .build();

      SearchResults baselineResults = baselineIndex.search(baselineQuery);

      // 2. Test Scalar 8-bit quantization
      CuVSQuantizer scalar8BitQuantizer = new Scalar8BitQuantizer(resources, trainingDataset);

      CagraIndex quantizedIndex =
          CagraIndex.newBuilder(resources)
              .withDataset(trainingDataset)
              .withQuantizer(scalar8BitQuantizer)
              .withIndexParams(indexParams)
              .build();

      CagraQuery quantizedQuery =
          CagraQuery.newBuilder()
              .withQueryVectors(queries)
              .withQuantizer(scalar8BitQuantizer)
              .withSearchParams(searchParams)
              .withTopK(10)
              .build();

      SearchResults quantizedResults = quantizedIndex.search(quantizedQuery);

      // 3. Calculate and verify recall
      double recall = calculateRecall(baselineResults, quantizedResults, 5); // Check recall@5
      log.info("Scalar 8-bit quantization recall@5: {}", recall);

      // Expect reasonable recall (should be > 0.6 for 8-bit quantization)
      assertTrue("Scalar 8-bit quantization should maintain reasonable recall", recall > 0.6);

      // 4. Test Binary quantization (expect lower recall)
      CuVSQuantizer binaryQuantizer = new BinaryQuantizer(resources);

      CagraIndex binaryIndex =
          CagraIndex.newBuilder(resources)
              .withDataset(trainingDataset)
              .withQuantizer(binaryQuantizer)
              .withIndexParams(indexParams)
              .build();

      CagraQuery binaryQuery =
          CagraQuery.newBuilder()
              .withQueryVectors(queries)
              .withQuantizer(binaryQuantizer)
              .withSearchParams(searchParams)
              .withTopK(10)
              .build();

      SearchResults binaryResults = binaryIndex.search(binaryQuery);

      double binaryRecall = calculateRecall(baselineResults, binaryResults, 5);
      log.info("Binary quantization recall@5: {}", binaryRecall);

      // Binary quantization should have lower recall than scalar quantization
      assertTrue("Binary quantization recall should be lower than scalar", binaryRecall < recall);
      assertTrue("Binary quantization should still have some recall", binaryRecall > 0.2);

      // 5. Verify quantization actually happened
      assertTrue("Scalar query should be quantized", quantizedQuery.hasQuantizedQueries());
      assertTrue("Binary query should be quantized", binaryQuery.hasQuantizedQueries());
      assertEquals("Scalar quantized data should be 8-bit", 8, quantizedQuery.getQueryPrecision());
      assertEquals("Binary quantized data should be 8-bit", 8, binaryQuery.getQueryPrecision());

      // Cleanup
      baselineIndex.destroyIndex();
      quantizedIndex.destroyIndex();
      binaryIndex.destroyIndex();
      scalar8BitQuantizer.close();
      binaryQuantizer.close();
      trainingDataset.close();
    }
  }

  private static float[][] createLargerSampleData() {
    // Create a larger dataset for meaningful recall testing
    return new float[][] {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f},
      {0.1f, 0.2f},
      {0.3f, 0.4f},
      {0.5f, 0.6f},
      {0.7f, 0.8f},
      {0.11f, 0.21f},
      {0.31f, 0.41f},
      {0.51f, 0.61f},
      {0.71f, 0.81f},
      {0.12f, 0.22f},
      {0.32f, 0.42f},
      {0.52f, 0.62f},
      {0.72f, 0.82f}
      // Add more vectors as needed
    };
  }

  private double calculateRecall(SearchResults baseline, SearchResults quantized, int k) {
    List<Map<Integer, Float>> baselineResults = baseline.getResults();
    List<Map<Integer, Float>> quantizedResults = quantized.getResults();

    double totalRecall = 0.0;

    for (int i = 0; i < baselineResults.size(); i++) {
      Set<Integer> baselineTopK =
          baselineResults.get(i).keySet().stream().limit(k).collect(Collectors.toSet());

      Set<Integer> quantizedTopK =
          quantizedResults.get(i).keySet().stream().limit(k).collect(Collectors.toSet());

      // Calculate intersection
      Set<Integer> intersection = new HashSet<>(baselineTopK);
      intersection.retainAll(quantizedTopK);

      double queryRecall = (double) intersection.size() / k;
      totalRecall += queryRecall;
    }

    return totalRecall / baselineResults.size();
  }
}
