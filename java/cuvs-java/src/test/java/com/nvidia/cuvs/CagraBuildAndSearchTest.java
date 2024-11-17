/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.common.CuVSResources;
import com.nvidia.cuvs.common.SearchResults;

public class CagraBuildAndSearchTest {
  
  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  /**
   * A basic test that checks the whole flow - from indexing to search.
   * 
   * @throws Throwable
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {

    // Sample data and query
    float[][] dataset = {{ 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f }, { 0.6673192f, 0.10993068f }};
    Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
    float[][] queries = {{ 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f }, { 0.05198065f, 0.5789965f }};
    
    // Expected search results
    List<Map<Integer, Float>> expectedResults = Arrays.asList(
        Map.of(3, 0.038782578f, 2, 0.3590463f, 0, 0.83774555f),
        Map.of(0, 0.12472608f, 2, 0.21700792f, 1, 0.31918612f), 
        Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f), 
        Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.5986642f));

    // Create resource
    CuVSResources resources = new CuVSResources();

    // Configure index parameters
    CagraIndexParams indexParams = new CagraIndexParams.Builder()
        .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
        .build();

    // Create the index with the dataset
    CagraIndex index = new CagraIndex.Builder(resources)
        .withDataset(dataset)
        .withIndexParams(indexParams)
        .build();

    // Saving the index on to the disk.
    String indexFileName = UUID.randomUUID().toString() + ".cag";
    index.serialize(new FileOutputStream(indexFileName));

    // Loading a CAGRA index from disk.
    File indexFile = new File(indexFileName);
    InputStream inputStream = new FileInputStream(indexFile);
    CagraIndex loadedIndex = new CagraIndex.Builder(resources)
        .from(inputStream)
        .build();
    
    // Configure search parameters
    CagraSearchParams searchParams = new CagraSearchParams.Builder()
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
    assertEquals(expectedResults, results.getResults(), "Results different than expected");

    // Search from deserialized index
    results = loadedIndex.search(cuvsQuery);
    
    // Check results
    log.info(results.getResults().toString());
    assertEquals(expectedResults, results.getResults(), "Results different than expected");

    // Cleanup
    if (indexFile.exists()) {
      indexFile.delete();
    }
  }
}