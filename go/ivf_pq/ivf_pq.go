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
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

type ivfPqIndex struct {
	index   C.cuvsIvfPqIndex_t
	trained bool
}

func CreateIndex(params *indexParams, dataset *cuvs.Tensor[float32]) (*ivfPqIndex, error) {

	index := (C.cuvsIvfPqIndex_t)(C.malloc(C.size_t(unsafe.Sizeof(C.cuvsIvfPqIndex{}))))
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &ivfPqIndex{index: index}, nil
}

func BuildIndex[T any](Resources cuvs.Resource, params *indexParams, dataset *cuvs.Tensor[T], index *ivfPqIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

func (index *ivfPqIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}

func SearchIndex[T any](Resources cuvs.Resource, params *searchParams, index *ivfPqIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[T]) error {

	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))

}
