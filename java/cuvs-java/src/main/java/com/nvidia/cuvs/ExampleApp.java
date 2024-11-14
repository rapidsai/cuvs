package com.nvidia.cuvs;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Map;

import com.nvidia.cuvs.cagra.CagraIndex;
import com.nvidia.cuvs.cagra.CagraIndexParams;
import com.nvidia.cuvs.cagra.CagraIndexParams.CuvsCagraGraphBuildAlgo;
import com.nvidia.cuvs.cagra.CagraSearchParams;
import com.nvidia.cuvs.cagra.CuVSQuery;
import com.nvidia.cuvs.cagra.CuVSResources;
import com.nvidia.cuvs.cagra.SearchResult;

public class ExampleApp {
  public static void main(String[] args) throws Throwable {

    // Sample data and query
    float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f } };
    Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
    float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f } };

    CuVSResources res = new CuVSResources();

    CagraIndexParams cagraIndexParams = new CagraIndexParams.Builder()
        .withIntermediateGraphDegree(10)
        .withBuildAlgo(CuvsCagraGraphBuildAlgo.IVF_PQ)
        .build();

    CagraSearchParams cagraSearchParams = new CagraSearchParams
        .Builder()
        .build();

    // Creating a new CAGRA index
    CagraIndex index = new CagraIndex.Builder(res)
        .withDataset(dataset)
        .withIndexParams(cagraIndexParams)
        .build();

    // Saving the index on to the disk.
    index.serialize(new FileOutputStream("abc.cag"));

    // Loading a CAGRA index from disk.
    InputStream fin = new FileInputStream(new File("abc.cag"));
    CagraIndex index2 = new CagraIndex.Builder(res)
        .from(fin)
        .build();

    // Query
    CuVSQuery query = new CuVSQuery.Builder()
        .withTopK(1)
        .withSearchParams(cagraSearchParams)
        .withQueryVectors(queries)
        .withMapping(map)
        .build();

    // Search
    SearchResult rslt = index.search(query);
    System.out.println(rslt.getResults());

    // Search from de-serialized index
    SearchResult rslt2 = index2.search(query);
    System.out.println(rslt2.getResults());

  }
}
