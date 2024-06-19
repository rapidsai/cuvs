package common

import (
	"math/rand"
	"testing"
	"time"
	"unsafe"
)

func TestBruteForce(t *testing.T) {

	resource := NewResource(nil)

	rand.Seed(time.Now().UnixNano())

	NDataPoints := 16
	NFeatures := 8

	TestDataset := make([]float32, NDataPoints*NFeatures)
	for index := range TestDataset {
		TestDataset[index] = rand.Float32()
		// TestDataset[index] = float32(index)
	}

	dataset := NewManagedTensor(true, []int{NDataPoints, NFeatures}, TestDataset, false)

	index := CreateIndex()
	// use the first 4 points from the dataset as queries : will test that we get them back
	// as their own nearest neighbor

	NQueries := 4
	K := 4
	queries := NewManagedTensor(true, []int{NQueries, NFeatures}, TestDataset[:(NQueries*NFeatures)], false)
	neighbors := NewManagedTensor(true, []int{NQueries, K}, make([]int64, NQueries*K), true)
	distances := NewManagedTensor(true, []int{NQueries, K}, make([]float32, NQueries*K), false)

	ToDevice(neighbors, &resource)
	ToDevice(distances, &resource)
	ToDevice(dataset, &resource)

	BuildIndex(resource.resource, dataset, "L2Expanded", 2.0, index)
	Sync(resource.resource)

	ToDevice(queries, &resource)

	index.trained = true

	SearchIndex(resource.resource, *index, queries, neighbors, distances)

	ToHost(neighbors, &resource)
	ToHost(distances, &resource)

	Sync(resource.resource)

	p := (*int64)(unsafe.Pointer(uintptr(neighbors.dl_tensor.data) + uintptr(K*8*3)))

	d := (*float32)(distances.dl_tensor.data)

	println(*p)

	println(*d)

}
