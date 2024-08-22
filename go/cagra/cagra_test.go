package cagra

import (
	"math/rand"
	"testing"
	"time"

	cuvs "github.com/rapidsai/cuvs/go"
)

func TestCagra(t *testing.T) {

	resource, _ := cuvs.NewResource(nil)

	rand.Seed(time.Now().UnixNano())

	NDataPoints := 1024
	NFeatures := 16

	TestDataset := make([][]float32, NDataPoints)
	for i := range TestDataset {
		TestDataset[i] = make([]float32, NFeatures)
		for j := range TestDataset[i] {
			TestDataset[i][j] = rand.Float32()
		}
	}

	ExtendDataPoints := 64
	ExtendDataset := make([][]float32, ExtendDataPoints)
	for i := range ExtendDataset {
		ExtendDataset[i] = make([]float32, NFeatures)
		for j := range ExtendDataset[i] {
			ExtendDataset[i][j] = rand.Float32()
		}
	}

	ExtendReturnEmptyDataset := make([][]float32, NDataPoints+ExtendDataPoints)
	for i := range ExtendReturnEmptyDataset {
		ExtendReturnEmptyDataset[i] = make([]float32, NFeatures)
		// for j := range ExtendReturnEmptyDataset[i] {
		// 	ExtendReturnEmptyDataset[i][j] = rand.Float32()
		// }
	}

	dataset, _ := cuvs.NewTensor(true, TestDataset)

	extend_dataset, _ := cuvs.NewTensor(true, ExtendDataset)

	extend_return_dataset, _ := cuvs.NewTensor(true, ExtendReturnEmptyDataset)

	// CompressionParams, _ := CreateCompressionParams()

	IndexParams, err := CreateIndexParams()

	// IndexParams.SetCompression(CompressionParams)

	if err != nil {
		panic(err)
	}

	index, _ := CreateIndex()
	defer index.Close()
	// use the first 4 points from the dataset as queries : will test that we get them back
	// as their own nearest neighbor

	NQueries := 4
	K := 10
	queries, _ := cuvs.NewTensor(true, TestDataset[:NQueries])
	NeighborsDataset := make([][]uint32, NQueries)
	for i := range NeighborsDataset {
		NeighborsDataset[i] = make([]uint32, K)
	}
	DistancesDataset := make([][]float32, NQueries)
	for i := range DistancesDataset {
		DistancesDataset[i] = make([]float32, K)
	}
	neighbors, _ := cuvs.NewTensor(true, NeighborsDataset)
	distances, _ := cuvs.NewTensor(true, DistancesDataset)
	println("hello")

	_, todeviceerr := neighbors.ToDevice(&resource)
	if todeviceerr != nil {
		println(todeviceerr)
	}
	distances.ToDevice(&resource)
	dataset.ToDevice(&resource)
	extend_dataset.ToDevice(&resource)
	extend_return_dataset.ToDevice(&resource)

	err = BuildIndex(resource, IndexParams, &dataset, index)
	if err != nil {
		// println(err.Error())
		panic(err)
	}
	resource.Sync()
	ExtendParams, err := CreateExtendParams()
	if err != nil {
		panic(err)
	}
	err = ExtendIndex(resource, ExtendParams, &extend_dataset, &extend_return_dataset, index)
	if err != nil {
		// println(err.Error())
		panic(err)
	}

	resource.Sync()

	queries.ToDevice(&resource)

	SearchParams, err := CreateSearchParams()

	if err != nil {
		panic(err)
	}

	err = SearchIndex(resource, SearchParams, index, &queries, &neighbors, &distances)
	if err != nil {
		panic(err)
	}

	neighbors.ToHost(&resource)
	distances.ToHost(&resource)

	resource.Sync()

	println("extend return dataset: ------------------------")

	extend_return_dataset.ToHost(&resource)
	arr_extend_return_dataset, _ := extend_return_dataset.GetArray()
	for i := range arr_extend_return_dataset {
		println(arr_extend_return_dataset[i][0])
		// if arr_extend_return_dataset[i][0] != float32(i) {
		// 	t.Error("wrong neighbor, expected", i, "got", arr_extend_return_dataset[i][0])
		// }
	}

	// p := (*int64)(unsafe.Pointer(uintptr(neighbors.c_tensor.dl_tensor.data) + uintptr(K*8*3)))
	arr, _ := neighbors.GetArray()
	for i := range arr {
		println(arr[i][0])
		if arr[i][0] != uint32(i) {
			t.Error("wrong neighbor, expected", i, "got", arr[i][0])
		}
	}

	// arr_dist, _ := distances.GetArray()
	// for i := range arr_dist {
	// 	if arr_dist[i][0] >= float32(0.001) || arr_dist[i][0] <= float32(-0.001) {
	// 		t.Error("wrong distance, expected", float32(i), "got", arr_dist[i][0])
	// 	}
	// }

}
