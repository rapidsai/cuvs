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
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;

public class HnswExample {

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

      HnswIndexParams hnswIndexParams = new HnswIndexParams.Builder()
          .withVectorDimension(2)
          .build();
      InputStream inputStreamHNSW = new FileInputStream(hnswIndexFileName);
      File hnswIndexFile = new File(hnswIndexFileName);

      HnswIndex hnswIndex = HnswIndex.newBuilder(resources)
          .from(inputStreamHNSW)
          .withIndexParams(hnswIndexParams)
          .build();

      HnswSearchParams hnswSearchParams = new HnswSearchParams.Builder()
          .build();

      HnswQuery hnswQuery = new HnswQuery.Builder()
          .withQueryVectors(queries)
          .withSearchParams(hnswSearchParams)
          .withTopK(3)
          .build();

      SearchResults results = hnswIndex.search(hnswQuery);

      // Check results
      log.info(results.getResults().toString());

      // Cleanup
      if (hnswIndexFile.exists()) {
        hnswIndexFile.delete();
      }
      index.destroyIndex();
      hnswIndex.destroyIndex();
    }
  }
}
