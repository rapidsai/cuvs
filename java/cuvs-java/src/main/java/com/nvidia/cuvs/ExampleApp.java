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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.cagra.CagraIndex;
import com.nvidia.cuvs.cagra.CagraIndexParams;
import com.nvidia.cuvs.cagra.CagraIndexParams.CuvsCagraGraphBuildAlgo;
import com.nvidia.cuvs.cagra.CagraSearchParams;
import com.nvidia.cuvs.cagra.CuVSQuery;
import com.nvidia.cuvs.cagra.CuVSResources;
import com.nvidia.cuvs.cagra.SearchResult;

public class ExampleApp {
  
  private static Logger logger = LoggerFactory.getLogger(ExampleApp.class);

  public static void main(String[] args) throws Throwable {

    // Sample data and query
    float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f } };
    Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
    float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f } };

    CuVSResources cuvsResources = new CuVSResources();

    CagraIndexParams cagraIndexParameters = new CagraIndexParams
        .Builder()
        .withIntermediateGraphDegree(10)
        .withCuvsCagraGraphBuildAlgo(CuvsCagraGraphBuildAlgo.NN_DESCENT)
        .build();

    // Creating a new CAGRA index
    CagraIndex cagraIndex = new CagraIndex
        .Builder(cuvsResources)
        .withDataset(dataset)
        .withIndexParams(cagraIndexParameters)
        .build();

    // Saving the CAGRA index on to the disk.
    File indexFile = new File("sample_index.cag");
    cagraIndex.serialize(new FileOutputStream(indexFile));

    // Loading a CAGRA index from disk.
    InputStream fileInputStream = new FileInputStream(indexFile);
    CagraIndex deserializedIndex = new CagraIndex
        .Builder(cuvsResources)
        .from(fileInputStream)
        .build();

    CagraSearchParams cagraSearchParameters = new CagraSearchParams
        .Builder()
        .build();

    // Query
    CuVSQuery cuvsQuery = new CuVSQuery
        .Builder()
        .withTopK(3)
        .withSearchParams(cagraSearchParameters)
        .withQueryVectors(queries)
        .withMapping(map)
        .build();

    // Search
    SearchResult searchResult = cagraIndex.search(cuvsQuery);
    logger.info(searchResult.getResults().toString());

    // Search from deserialized index
    SearchResult searchResultFromDeserializedIndex = deserializedIndex.search(cuvsQuery);
    logger.info(searchResultFromDeserializedIndex.getResults().toString());
    
    // Cleanup
    if (indexFile.exists()) {
      indexFile.delete();
    }
  }
}