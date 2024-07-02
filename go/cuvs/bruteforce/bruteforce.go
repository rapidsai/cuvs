package bruteforce

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

	err := common.CheckCuvs(common.CuvsError(C.cuvsBruteForceIndexCreate(&index)))

	if err != nil {
		return nil, err
	}

	return &Index{index: index, trained: false}, nil

}

func (index *Index) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsBruteForceIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}

func BuildIndex[T any](Resources common.Resource, Dataset *common.Tensor[T], metric string, metric_arg float32, index *Index) error {

	// if Dataset.C_tensor.dl_tensor.device.device_type != C.kDLCUDA {
	// 	return errors.New("dataset must be on GPU")
	// }

	CMetric := C.cuvsDistanceType(0)

	switch metric {
	case "L2Expanded":
		CMetric = C.L2Expanded
	default:
		return errors.New("unsupported metric")
	}

	err := common.CheckCuvs(common.CuvsError(C.cuvsBruteForceBuild(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(Dataset.C_tensor)), CMetric, C.float(metric_arg), index.index)))
	if err != nil {
		return err
	}
	index.trained = true

	println("build done")

	return nil

}

func SearchIndex[T any](resources common.Resource, index Index, queries *common.Tensor[T], neighbors *common.Tensor[int64], distances *common.Tensor[T]) error {

	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return common.CheckCuvs(common.CuvsError(C.cuvsBruteForceSearch(C.ulong(resources.Resource), index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))

}
