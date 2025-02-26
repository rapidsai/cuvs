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

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CuVSResources;

public class BruteForceExample {

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

      // Create a query object with the query vectors
      BruteForceQuery cuvsQuery = new BruteForceQuery.Builder()
          .withTopK(3)
          .withQueryVectors(queries)
          .build();

      // Set index parameters
      BruteForceIndexParams indexParams = new BruteForceIndexParams.Builder()
          .build();

      // Create the index with the dataset
      BruteForceIndex index = BruteForceIndex.newBuilder(resources)
          .withDataset(dataset)
          .withIndexParams(indexParams)
          .build();

      // Saving the index on to the disk.
      String indexFileName = UUID.randomUUID().toString() + ".bf";
      index.serialize(new FileOutputStream(indexFileName));

      // Loading a BRUTEFORCE index from disk.
      File indexFile = new File(indexFileName);
      InputStream inputStream = new FileInputStream(indexFile);
      BruteForceIndex loadedIndex = BruteForceIndex.newBuilder(resources)
          .from(inputStream)
          .build();

      // Perform the search
      SearchResults resultsFromLoadedIndex = loadedIndex.search(cuvsQuery);

      // Check results
      log.info(resultsFromLoadedIndex.getResults().toString());

      // Perform the search
      SearchResults results = index.search(cuvsQuery);

      // Check results
      log.info(results.getResults().toString());

      // Cleanup
      index.destroyIndex();
      loadedIndex.destroyIndex();

      if (indexFile.exists()) {
        indexFile.delete();
      }
    }
  }
}
