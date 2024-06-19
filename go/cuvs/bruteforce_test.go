package common

import (
	"testing"
)

func TestBruteForce(t *testing.T) {

	resource := NewResource(nil)

	dataset := NewManagedTensor(true, []int{1, 2}, []float32{1, 2}, false)
	bytes := GetBytes(dataset)
	println("size:")
	println(bytes)
	println(dataset.dl_tensor.dtype.code)

	index := CreateIndex()

	BuildIndex(resource.resource, dataset, "L2Expanded", 2.0, index)
	println("got here")

	println("index created")

	index.trained = true

	queries := NewManagedTensor(true, []int{1, 2}, []float32{1, 2}, false)

	neighbors := NewManagedTensor(true, []int{1, 1}, []int64{3, 3}, true)

	// distances := NewManagedTensor(true, []int{1, 1}, []float32{1, 4}, false)

	println("tensors created")

	println(queries.dl_tensor.data)

	queries_device := ToDevice(queries, &resource)

	println("queries tensor transferred")

	// neighbors_device := ToDevice(neighbors, &resource)

	// distances_device := ToDevice(distances, &resource)

	println(queries_device.dl_tensor.data)

	println("tensors on device")

	// SearchIndex(resource.resource, *index, queries_device, neighbors_device, distances_device)

	println("search done")

	p := (*float32)(neighbors.dl_tensor.data)

	println(*p)

}
