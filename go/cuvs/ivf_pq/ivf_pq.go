package ivf_pq

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

type IvfPqIndex struct {
	index   C.cuvsIvfPqIndex_t
	trained bool
}

func CreateIndex(params *IndexParams, dataset *common.Tensor[float32]) (*IvfPqIndex, error) {

	index := (C.cuvsIvfPqIndex_t)(C.malloc(C.size_t(unsafe.Sizeof(C.cuvsIvfPqIndex{}))))
	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfPqIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &IvfPqIndex{index: index}, nil
}

type ManagedTensor = *C.DLManagedTensor

func BuildIndex[T any](Resources common.Resource, params *IndexParams, dataset *common.Tensor[T], index *IvfPqIndex) error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfPqBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

func (index *IvfPqIndex) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfPqIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}

func SearchIndex[T any](Resources common.Resource, params *SearchParams, index *IvfPqIndex, queries *common.Tensor[T], neighbors *common.Tensor[int64], distances *common.Tensor[T]) error {

	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return common.CheckCuvs(common.CuvsError(C.cuvsIvfPqSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))

}
