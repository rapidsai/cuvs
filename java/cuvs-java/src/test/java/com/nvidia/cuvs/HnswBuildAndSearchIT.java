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

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;
import com.nvidia.cuvs.SearchResults;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;

public class HnswBuildAndSearchIT {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

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
        Map.of(3, 0.038782578f, 2, 0.35904628f, 0, 0.8377455f),
        Map.of(0, 0.12472608f, 2, 0.21700794f, 1, 0.31918612f),
        Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f),
        Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.59866416f)
      );

    for (int j = 0; j < 10; j++) {

      try (CuVSResources resources = CuVSResources.create()) {

        // Configure index parameters
        CagraIndexParams indexParams = new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
            .withGraphDegree(64)
            .withIntermediateGraphDegree(128)
            .withNumWriterThreads(32)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

        // Create the index with the dataset
        CagraIndex index = CagraIndex.newBuilder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        // Saving the HNSW index on to the disk.
        String hnswIndexFileName = UUID.randomUUID().toString() + ".hnsw";
        index.serializeToHNSW(new FileOutputStream(hnswIndexFileName));

        HnswIndexParams hnswIndexParams = new HnswIndexParams.Builder(resources)
            .withVectorDimension(2)
            .build();
        InputStream inputStreamHNSW = new FileInputStream(hnswIndexFileName);
        File hnswIndexFile = new File(hnswIndexFileName);

        HnswIndex hnswIndex = HnswIndex.newBuilder(resources)
            .from(inputStreamHNSW)
            .withIndexParams(hnswIndexParams)
            .build();

        HnswSearchParams hnswSearchParams = new HnswSearchParams.Builder(resources)
            .build();

        HnswQuery hnswQuery = new HnswQuery.Builder()
            .withMapping(map)
            .withQueryVectors(queries)
            .withSearchParams(hnswSearchParams)
            .withTopK(3)
            .build();

        SearchResults results = hnswIndex.search(hnswQuery);

        // Check results
        log.info(results.getResults().toString());
        assertEquals(expectedResults, results.getResults());

        // Cleanup
        if (hnswIndexFile.exists()) {
          hnswIndexFile.delete();
        }
        index.destroyIndex();
        hnswIndex.destroyIndex();
      }
    }
  }
}
