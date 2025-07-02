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

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.LongToIntFunction;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HnswBuildAndSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  // Sample data and query
  private static final float[][] dataset = {
    {0.74021935f, 0.9209938f},
    {0.03902049f, 0.9689629f},
    {0.92514056f, 0.4463501f},
    {0.6673192f, 0.10993068f}
  };

  private static final float[][] queries = {
    {0.48216683f, 0.0428398f},
    {0.5084142f, 0.6545497f},
    {0.51260436f, 0.2643005f},
    {0.05198065f, 0.5789965f}
  };

  private static void indexAndQueryOnce(
      CuVSResources resources, HnswQuery hnswQuery, List<Map<Integer, Float>> expectedResults)
      throws Throwable {
    // Configure index parameters
    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(64)
            .withIntermediateGraphDegree(128)
            .withNumWriterThreads(32)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    // Create the index with the dataset
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(indexParams).build();

    // Saving the HNSW index on to the disk.
    String hnswIndexFileName = UUID.randomUUID() + ".hnsw";
    var hnswIndexPath = Path.of(hnswIndexFileName);
    try {
      try (var outputStream = Files.newOutputStream(hnswIndexPath)) {
        index.serializeToHNSW(outputStream);
      }

      HnswIndexParams hnswIndexParams =
          new HnswIndexParams.Builder().withVectorDimension(2).build();
      try (var inputStreamHNSW = Files.newInputStream(hnswIndexPath)) {
        var hnswIndex =
            HnswIndex.newBuilder(resources)
                .from(inputStreamHNSW)
                .withIndexParams(hnswIndexParams)
                .build();

        SearchResults results = hnswIndex.search(hnswQuery);

        // Check results
        log.info(results.getResults().toString());
        checkResults(expectedResults, results.getResults());

        // Cleanup
        hnswIndex.destroyIndex();
      }
    } finally {
      index.destroyIndex();
      Files.deleteIfExists(hnswIndexPath);
    }
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {
    // Expected search results
    final List<Map<Integer, Float>> expectedResults =
        Arrays.asList(
            Map.of(3, 0.038782578f, 2, 0.35904628f, 0, 0.8377455f),
            Map.of(0, 0.12472608f, 2, 0.21700794f, 1, 0.31918612f),
            Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f),
            Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.59866416f));

    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CuVSResources.create()) {
        HnswQuery hnswQuery =
            new HnswQuery.Builder()
                .withMapping(SearchResults.IDENTITY_MAPPING)
                .withQueryVectors(queries)
                .withSearchParams(new HnswSearchParams.Builder().build())
                .withTopK(3)
                .build();
        indexAndQueryOnce(resources, hnswQuery, expectedResults);
      }
    }
  }

  @Test
  public void testIndexingAndSearchingWithFunctionMapping() throws Throwable {
    // Expected search results
    final List<Map<Integer, Float>> expectedResults =
        Arrays.asList(
            Map.of(0, 0.038782578f, 3, 0.35904628f, 1, 0.8377455f),
            Map.of(1, 0.12472608f, 3, 0.21700794f, 2, 0.31918612f),
            Map.of(0, 0.047766715f, 3, 0.20332818f, 1, 0.48305473f),
            Map.of(2, 0.15224178f, 1, 0.59063464f, 0, 0.59866416f));
    LongToIntFunction rotate = l -> (int) ((l + 1) % dataset.length);

    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CuVSResources.create()) {
        HnswQuery hnswQuery =
            new HnswQuery.Builder()
                .withQueryVectors(queries)
                .withMapping(rotate)
                .withSearchParams(new HnswSearchParams.Builder().build())
                .withTopK(3)
                .build();
        indexAndQueryOnce(resources, hnswQuery, expectedResults);
      }
    }
  }

  @Test
  public void testIndexingAndSearchingWithListMapping() throws Throwable {
    // Expected search results
    final List<Map<Integer, Float>> expectedResults =
        Arrays.asList(
            Map.of(1, 0.038782578f, 2, 0.35904628f, 4, 0.8377455f),
            Map.of(4, 0.12472608f, 2, 0.21700794f, 3, 0.31918612f),
            Map.of(1, 0.047766715f, 2, 0.20332818f, 4, 0.48305473f),
            Map.of(3, 0.15224178f, 4, 0.59063464f, 1, 0.59866416f));
    var mappings = List.of(4, 3, 2, 1);
    LongToIntFunction rotate = SearchResults.mappingsFromList(mappings);

    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CuVSResources.create()) {
        HnswQuery hnswQuery =
            new HnswQuery.Builder()
                .withQueryVectors(queries)
                .withMapping(rotate)
                .withSearchParams(new HnswSearchParams.Builder().build())
                .withTopK(3)
                .build();
        indexAndQueryOnce(resources, hnswQuery, expectedResults);
      }
    }
  }
}
