package cagra

// #include <stdio.h>
// #include <stdlib.h>
// #include <dlpack/dlpack.h>
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
	"rapidsai/cuvs"
	"unsafe"
)

type cagraIndex struct {
	index   C.cuvsCagraIndex_t
	trained bool
}

func CreateIndex(params *indexParams, dataset *cuvs.Tensor[float32]) (*cagraIndex, error) {

	index := (C.cuvsCagraIndex_t)(C.malloc(C.size_t(unsafe.Sizeof(C.cuvsCagraIndex{}))))
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &cagraIndex{index: index}, nil
}

func BuildIndex[T any](Resources cuvs.Resource, params *indexParams, dataset *cuvs.Tensor[T], index *cagraIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

func (index *cagraIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}

func SearchIndex[T any](Resources cuvs.Resource, params *searchParams, index *cagraIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[uint32], distances *cuvs.Tensor[T]) error {

	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))

}
