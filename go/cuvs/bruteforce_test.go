package common

import (
	"math/rand"
	"testing"
	"time"
)

func TestBruteForce(t *testing.T) {

	resource, _ := NewResource(nil)

	rand.Seed(time.Now().UnixNano())

	NDataPoints := 16
	NFeatures := 8

	TestDataset := make([][]float32, NDataPoints)
	for i := range TestDataset {
		TestDataset[i] = make([]float32, NFeatures)
		for j := range TestDataset[i] {
			TestDataset[i][j] = rand.Float32()
		}
	}

	dataset, _ := NewTensor(true, TestDataset)

	index, _ := CreateIndex()
	defer index.Close()
	// use the first 4 points from the dataset as queries : will test that we get them back
	// as their own nearest neighbor

	NQueries := 4
	K := 4
	queries, _ := NewTensor(true, TestDataset[:NQueries])
	NeighborsDataset := make([][]int64, NQueries)
	for i := range NeighborsDataset {
		NeighborsDataset[i] = make([]int64, K)
	}
	DistancesDataset := make([][]float32, NQueries)
	for i := range DistancesDataset {
		DistancesDataset[i] = make([]float32, K)
	}
	neighbors, _ := NewTensor(true, NeighborsDataset)
	distances, _ := NewTensor(true, DistancesDataset)

	neighbors.ToDevice(&resource)
	distances.ToDevice(&resource)
	dataset.ToDevice(&resource)

	BuildIndex(resource.resource, &dataset, "L2Expanded", 2.0, index)
	Sync(resource.resource)

	queries.ToDevice(&resource)

	SearchIndex(resource.resource, *index, &queries, &neighbors, &distances)

	neighbors.ToHost(&resource)
	distances.ToHost(&resource)

	Sync(resource.resource)

	// p := (*int64)(unsafe.Pointer(uintptr(neighbors.c_tensor.dl_tensor.data) + uintptr(K*8*3)))
	arr, _ := neighbors.GetArray()
	for i := range arr {
		println(arr[i][0])
		if arr[i][0] != int64(i) {
			t.Error("wrong neighbor, expected", i, "got", arr[i][0])
		}
	}

	arr_dist, _ := distances.GetArray()
	for i := range arr_dist {
		if arr_dist[i][0] >= float32(0.001) || arr_dist[i][0] <= float32(-0.001) {
			t.Error("wrong distance, expected", float32(i), "got", arr_dist[i][0])
		}
	}

}
