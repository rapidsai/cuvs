package common

import (
	"testing"
)

func TestBruteForce(t *testing.T) {

	resource := NewResource(nil)

	dataset := NewManagedTensor(true, []int{1, 1}, []float32{1, 2})

	println(dataset.dl_tensor.dtype.code)

	index := CreateIndex()

	BuildIndex(resource.resource, dataset, "L2Expanded", 2.0, index.index)

	index.trained = true

	queries := NewManagedTensor(true, []int{1, 1}, []float32{1, 2})

	neighbors := NewManagedTensor(true, []int{1, 1}, []float32{3, 3})

	distances := NewManagedTensor(true, []int{1, 1}, []float32{1, 4})

	SearchIndex(resource.resource, *index, queries, neighbors, distances)

	p := (*float32)(neighbors.dl_tensor.data)

	println(*p)

}
