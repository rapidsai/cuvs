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
	"unsafe"
)

type Index struct {
	index   C.cuvsBruteForceIndex_t
	trained bool
}

// func (index *Index) Close() {
// 	// C.free(index.index)
// }

func CreateIndex() *Index {

	index := (C.cuvsBruteForceIndex_t)(C.malloc(C.size_t(unsafe.Sizeof(C.cuvsBruteForceIndex{}))))

	// defer C.free(unsafe.Pointer(index))

	err := C.cuvsBruteForceIndexCreate(&index)

	CheckCuvs(err)

	return &Index{index: index, trained: false}

}

func DestroyIndex(index Index) {
	err := C.cuvsBruteForceIndexDestroy(index.index)
	CheckCuvs(err)

}

func BuildIndex(Resources C.cuvsResources_t, Dataset *C.DLManagedTensor, metric string, metric_arg float32, index *Index) {

	if Dataset.dl_tensor.device.device_type != C.kDLCUDA {
		panic("Dataset must be on GPU")
	}

	// Data := unsafe.Pointer(Dataset)

	// CheckCuvs(C.cuvsRMMAlloc(Resources, &Data, 2400))

	CMetric := C.cuvsDistanceType(0)

	switch metric {
	case "L2Expanded":
		CMetric = C.L2Expanded
	default:
		panic("Unsupported metric")
	}

	println(index.index.addr)

	CheckCuvs(C.cuvsBruteForceBuild(Resources, Dataset, CMetric, C.float(metric_arg), index.index))
	println(index.index.addr)

}

func SearchIndex(resources C.cuvsResources_t, index Index, queries *C.DLManagedTensor, neighbors *C.DLManagedTensor, distances *C.DLManagedTensor) {

	if !index.trained {
		panic("Index needs to be built before calling search.")
	}

	CheckCuvs(C.cuvsBruteForceSearch(resources, index.index, queries, neighbors, distances))

}
