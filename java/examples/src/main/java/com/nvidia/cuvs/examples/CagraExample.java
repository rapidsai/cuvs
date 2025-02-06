package com.nvidia.cuvs.examples;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.util.UUID;

import com.nvidia.cuvs.SearchResults;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSResources;

public class CagraExample {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  public static void main(String[] args) throws Throwable {

    // Sample data and query
    float[][] dataset = {
        { 0.74021935f, 0.9209938f },
        { 0.03902049f, 0.9689629f },
        { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f }
       };

    float[][] queries = {
        { 0.48216683f, 0.0428398f },
        { 0.5084142f, 0.6545497f },
        { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f }
       };

    try (CuVSResources resources = CuVSResources.create()) {

      // Configure index parameters
      CagraIndexParams indexParams = new CagraIndexParams.Builder()
          .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
          .withGraphDegree(1)
          .withIntermediateGraphDegree(2)
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
          .build();

      // Perform the search
      SearchResults results = index.search(cuvsQuery);

      // Check results
      log.info(results.getResults().toString());

      // Search from deserialized index
      results = loadedIndex.search(cuvsQuery);

      // Check results
      log.info(results.getResults().toString());

      // Cleanup
      if (indexFile.exists()) {
        indexFile.delete();
      }
      index.destroyIndex();
    }
  }
}
