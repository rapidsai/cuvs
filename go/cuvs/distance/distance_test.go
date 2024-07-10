package distance

import (
	"rapidsai/cuvs/cuvs/common"
	"testing"
	"time"

	"math/rand"
)

func TestDistance(t *testing.T) {
	resource, _ := common.NewResource(nil)

	rand.Seed(time.Now().UnixNano())

	NDataPoints := 256
	NFeatures := 16

	TestDataset := make([][]float32, NDataPoints)
	for i := range TestDataset {
		TestDataset[i] = make([]float32, NFeatures)
		for j := range TestDataset[i] {
			TestDataset[i][j] = rand.Float32()
		}
	}

	dataset, _ := common.NewTensor(true, TestDataset)

	DistancesDataset := make([][]float32, NDataPoints)
	for i := range DistancesDataset {
		DistancesDataset[i] = make([]float32, NDataPoints)
	}

	distances, _ := common.NewTensor(true, DistancesDataset)

	distances.ToDevice(&resource)
	dataset.ToDevice(&resource)

	PairwiseDistance(resource, &dataset, &dataset, &distances, L2, 0.0)

	distances.ToHost(&resource)

	resource.Sync()

	arr, _ := distances.GetArray()
	if arr[0][0] != 0.0 {
		t.Error("wrong distance, expected", 0.0, "got", arr[0][0])
	}

}
