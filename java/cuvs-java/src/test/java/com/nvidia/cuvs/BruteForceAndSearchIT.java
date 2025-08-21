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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.LongToIntFunction;
import org.junit.Before;
import org.junit.Test;

public class BruteForceAndSearchIT extends CuVSTestCase {

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
      CuVSResources resources, BruteForceQuery cuvsQuery, List<Map<Integer, Float>> expectedResults)
      throws Throwable {
    // Set index parameters
    BruteForceIndexParams indexParams =
        new BruteForceIndexParams.Builder().withNumWriterThreads(32).build();

    // Create the index with the dataset
    try (BruteForceIndex index =
        BruteForceIndex.newBuilder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build()) {

      // Saving the index on to the disk.
      String indexFileName = UUID.randomUUID() + ".bf";
      try (var outputStream = new FileOutputStream(indexFileName)) {
        index.serialize(outputStream);
      }

      // Loading a BRUTEFORCE index from disk.
      Path indexFile = Path.of(indexFileName);
      try (var inputStream = Files.newInputStream(indexFile);
          BruteForceIndex loadedIndex =
              BruteForceIndex.newBuilder(resources).from(inputStream).build()) {

        // search the loaded index
        SearchResults results = loadedIndex.search(cuvsQuery);
        checkResults(expectedResults, results.getResults());

        // search the first index
        results = index.search(cuvsQuery);
        checkResults(expectedResults, results.getResults());
      }
      Files.deleteIfExists(indexFile);
    }
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {
    // Expected search results
    final List<Map<Integer, Float>> expectedResults =
        List.of(
            Map.of(3, 0.038782537f, 2, 0.35904616f, 0, 0.83774555f),
            Map.of(0, 0.12472606f, 2, 0.21700788f, 1, 0.3191862f),
            Map.of(3, 0.047766685f, 2, 0.20332813f, 0, 0.48305476f),
            Map.of(1, 0.15224183f, 0, 0.5906347f, 3, 0.5986643f));
    for (int j = 0; j < 10; j++) {

      try (CuVSResources resources = CheckedCuVSResources.create()) {
        BruteForceQuery cuvsQuery =
            new BruteForceQuery.Builder(resources)
                .withTopK(3)
                .withQueryVectors(queries)
                .withMapping(SearchResults.IDENTITY_MAPPING)
                .build();
        indexAndQueryOnce(resources, cuvsQuery, expectedResults);
      }
    }
  }

  @Test
  public void testIndexingAndSearchingWithFiltering() throws Throwable {
    final BitSet prefilter = new BitSet();
    prefilter.set(0);
    prefilter.set(1);
    prefilter.clear(2);
    prefilter.set(3);

    final List<Map<Integer, Float>> expectedResultsWithFiltering =
        List.of(
            Map.of(0, 0.83774555f, 1, 1.0540828f, 3, 0.038782537f),
            Map.of(0, 0.12472606f, 1, 0.3191862f, 3, 0.32186073f),
            Map.of(0, 0.48305476f, 1, 0.7208309f, 3, 0.047766685f),
            Map.of(0, 0.5906347f, 1, 0.15224195f, 3, 0.5986643f));

    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CheckedCuVSResources.create()) {
        BruteForceQuery cuvsQuery =
            new BruteForceQuery.Builder(resources)
                .withTopK(3)
                .withQueryVectors(queries)
                .withPrefilters(
                    new BitSet[] {prefilter, prefilter, prefilter, prefilter}, dataset.length)
                .withMapping(SearchResults.IDENTITY_MAPPING)
                .build();
        indexAndQueryOnce(resources, cuvsQuery, expectedResultsWithFiltering);
      }
    }
  }

  @Test
  public void testIndexingAndSearchingWithFunctionMapping() throws Throwable {
    // Expected search results
    final List<Map<Integer, Float>> expectedResults =
        List.of(
            Map.of(0, 0.038782537f, 3, 0.35904616f, 1, 0.83774555f),
            Map.of(1, 0.12472606f, 3, 0.21700788f, 2, 0.3191862f),
            Map.of(0, 0.047766685f, 3, 0.20332813f, 1, 0.48305476f),
            Map.of(2, 0.15224183f, 1, 0.5906347f, 0, 0.5986643f));
    LongToIntFunction rotate = l -> (int) ((l + 1) % dataset.length);

    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CheckedCuVSResources.create()) {
        BruteForceQuery cuvsQuery =
            new BruteForceQuery.Builder(resources)
                .withTopK(3)
                .withQueryVectors(queries)
                .withMapping(rotate)
                .build();
        indexAndQueryOnce(resources, cuvsQuery, expectedResults);
      }
    }
  }

  @Test
  public void testIndexingAndSearchingWithListMapping() throws Throwable {
    // Expected search results
    final List<Map<Integer, Float>> expectedResults =
        List.of(
            Map.of(1, 0.038782537f, 2, 0.35904616f, 4, 0.83774555f),
            Map.of(4, 0.12472606f, 2, 0.21700788f, 3, 0.3191862f),
            Map.of(1, 0.047766685f, 2, 0.20332813f, 4, 0.48305476f),
            Map.of(3, 0.15224183f, 4, 0.5906347f, 1, 0.5986643f));
    var mappings = List.of(4, 3, 2, 1);
    LongToIntFunction rotate = SearchResults.mappingsFromList(mappings);

    for (int j = 0; j < 10; j++) {
      try (CuVSResources resources = CheckedCuVSResources.create()) {
        BruteForceQuery cuvsQuery =
            new BruteForceQuery.Builder(resources)
                .withTopK(3)
                .withQueryVectors(queries)
                .withMapping(rotate)
                .build();
        indexAndQueryOnce(resources, cuvsQuery, expectedResults);
      }
    }
  }
}
