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

package com.nvidia.cuvs.examples;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.common.CuVSResources;
import com.nvidia.cuvs.common.SearchResults;

public class CagraExample {
  
  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  public static void main(String[] args) throws Throwable {

    // Sample data and query
    float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f } };
    float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f } };

    // Allocate the resources
    CuVSResources resources = new CuVSResources();

    // Building a new CAGRA index
    CagraIndexParams indexParams = new CagraIndexParams.Builder()
        .withIntermediateGraphDegree(10)
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .build();
    CagraIndex index = new CagraIndex.Builder(resources)
        .withDataset(dataset)
        .withIndexParams(indexParams)
        .build();

    // Serializing the index to a file
    File indexFile = new File("cagra_index.cag");
    index.serialize(new FileOutputStream(indexFile));

    // Loading a CAGRA index from the file.
    InputStream fileInputStream = new FileInputStream(indexFile);
    CagraIndex loadedIndex = new CagraIndex.Builder(resources)
        .from(fileInputStream)
        .build();

    // Search
    CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
    CagraQuery query = new CagraQuery.Builder()
        .withTopK(3)
        .withSearchParams(searchParams)
        .withQueryVectors(queries)
        .build();
    SearchResults results = index.search(query);
    log.info(results.getResults().toString());

    // Search from loaded index
    results = loadedIndex.search(query);
    log.info(results.getResults().toString());
    
    // Cleanup
    if (indexFile.exists()) {
      indexFile.delete();
    }
  }
}