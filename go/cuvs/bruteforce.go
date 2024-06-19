package common

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
)

type Index struct {
	index   C.cuvsBruteForceIndex_t
	trained bool
}

// func (index *Index) Close() {
// 	// C.free(index.index)
// }

func CreateIndex() (*Index, error) {

	index := (C.cuvsBruteForceIndex_t)(C.malloc(C.size_t(unsafe.Sizeof(C.cuvsBruteForceIndex{}))))

	if index == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := CheckCuvs(C.cuvsBruteForceIndexCreate(&index))

	if err != nil {
		return nil, err
	}

	return &Index{index: index, trained: false}, nil

}

func (index *Index) Close() error {
	err := CheckCuvs(C.cuvsBruteForceIndexDestroy(index.index))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}

func BuildIndex[T any](Resources C.cuvsResources_t, Dataset *Tensor[T], metric string, metric_arg float32, index *Index) error {

	if Dataset.c_tensor.dl_tensor.device.device_type != C.kDLCUDA {
		return errors.New("dataset must be on GPU")
	}

	CMetric := C.cuvsDistanceType(0)

	switch metric {
	case "L2Expanded":
		CMetric = C.L2Expanded
	default:
		return errors.New("unsupported metric")
	}

	println(index.index.addr)

	err := CheckCuvs(C.cuvsBruteForceBuild(Resources, Dataset.c_tensor, CMetric, C.float(metric_arg), index.index))
	if err != nil {
		return err
	}
	index.trained = true

	return nil

}

func SearchIndex[T any](resources C.cuvsResources_t, index Index, queries *Tensor[T], neighbors *Tensor[int64], distances *Tensor[T]) error {

	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return CheckCuvs(C.cuvsBruteForceSearch(resources, index.index, queries.c_tensor, neighbors.c_tensor, distances.c_tensor))

}
