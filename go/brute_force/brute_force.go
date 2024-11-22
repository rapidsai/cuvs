package brute_force

// #include <cuvs/neighbors/brute_force.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

type bruteForceIndex struct {
	index   C.cuvsBruteForceIndex_t
	trained bool
}

func CreateIndex() (*bruteForceIndex, error) {
	var index C.cuvsBruteForceIndex_t

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
	return nil
}

func BuildIndex[T any](Resources cuvs.Resource, Dataset *cuvs.Tensor[T], metric cuvs.Distance, metric_arg float32, index *bruteForceIndex) error {
	CMetric, exists := cuvs.CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceBuild(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(Dataset.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg), index.index)))
	if err != nil {
		return err
	}
	index.trained = true

	return nil
}

func SearchIndex[T any](resources cuvs.Resource, index bruteForceIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[T]) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	prefilter := C.cuvsFilter{
		addr:  0,
		_type: C.NO_FILTER,
	}

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSearch(C.ulong(resources.Resource), index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), prefilter)))
}
