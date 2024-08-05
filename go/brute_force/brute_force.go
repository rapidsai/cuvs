package brute_force

// #include <cuda_runtime_api.h>
// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"
import (
	"errors"

	cuvs "github.com/rapidsai/cuvs/go"

	"unsafe"
)

type bruteForceIndex struct {
	index   C.cuvsBruteForceIndex_t
	trained bool
}

// func (index *Index) Close() {
// 	// C.free(index.index)
// }

func CreateIndex() (*bruteForceIndex, error) {

	index := (C.cuvsBruteForceIndex_t)(C.malloc(C.size_t(unsafe.Sizeof(C.cuvsBruteForceIndex{}))))

	if index == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexCreate(&index)))

	if err != nil {
		return nil, err
	}

	return &bruteForceIndex{index: index, trained: false}, nil

}

func (index *bruteForceIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}

func BuildIndex[T any](Resources cuvs.Resource, Dataset *cuvs.Tensor[T], metric cuvs.Distance, metric_arg float32, index *bruteForceIndex) error {

	// if Dataset.C_tensor.dl_tensor.device.device_type != C.kDLCUDA {
	// 	return errors.New("dataset must be on GPU")
	// }

	CMetric, exists := cuvs.CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceBuild(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(Dataset.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg), index.index)))
	if err != nil {
		return err
	}
	index.trained = true

	println("build done")

	return nil

}

func SearchIndex[T any](resources cuvs.Resource, index bruteForceIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[T]) error {

	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSearch(C.ulong(resources.Resource), index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))

}
