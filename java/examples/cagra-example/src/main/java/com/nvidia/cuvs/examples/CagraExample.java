package com.nvidia.cuvs.examples;

import java.lang.invoke.MethodHandles;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.common.SearchResults;

public class CagraExample {

	private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	public static void main(String[] args) throws Throwable {
		// Sample dataset and queries
		float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
				{ 0.6673192f, 0.10993068f } };
		float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
				{ 0.05198065f, 0.5789965f } };

		// Allocate the resources
		CuVSResources resources = new CuVSResources();

		// Create an index and a searcher
		CagraIndexParams indexParams = new CagraIndexParams.Builder(resources)
				.withIntermediateGraphDegree(10)
				.withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
				.build();
		CagraIndex index = new CagraIndex.Builder(resources)
				.withDataset(dataset)
				.withIndexParams(indexParams)
				.build();
		log.info("Indexing done!");

		// Search
		CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();
		CagraQuery query = new CagraQuery.Builder()
				.withTopK(2) // get only the top 2 items
				.withSearchParams(searchParams)
				.withQueryVectors(queries)
				.build();
		SearchResults results = index.search(query);

		// Print the results
		for (int i=0; i<results.getResults().size(); i++) {
			log.info("Query " + i + ": ");
			for (Integer docId: results.getResults().get(i).keySet()) {
				log.info("\tNeighbor ID: " + docId + ", distance from query point: " + results.getResults().get(i).get(docId));
			}
		}
	}
}
