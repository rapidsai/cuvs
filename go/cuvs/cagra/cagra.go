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
	"rapidsai/cuvs/cuvs/common"
	"unsafe"
)

type CagraIndex struct {
	index   C.cuvsCagraIndex_t
	trained bool
}

func CreateIndex(params *IndexParams, dataset *common.Tensor[float32]) (*CagraIndex, error) {

	index := (C.cuvsCagraIndex_t)(C.malloc(C.size_t(unsafe.Sizeof(C.cuvsCagraIndex{}))))
	err := common.CheckCuvs(common.CuvsError(C.cuvsCagraIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &CagraIndex{index: index}, nil
}

type ManagedTensor = *C.DLManagedTensor

func BuildIndex[T any](Resources common.Resource, params *IndexParams, dataset *common.Tensor[T], index *CagraIndex) error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsCagraBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

func (index *CagraIndex) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsCagraIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}

func SearchIndex[T any](Resources common.Resource, params *SearchParams, index *CagraIndex, queries *common.Tensor[T], neighbors *common.Tensor[uint32], distances *common.Tensor[T]) error {

	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return common.CheckCuvs(common.CuvsError(C.cuvsCagraSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))

}
