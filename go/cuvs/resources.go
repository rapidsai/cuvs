package common

// #cgo CFLAGS: -I/usr/local/cuda/include -I/home/ajit/miniforge3/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/home/ajit/miniforge3/lib -lcudart -lcudart -lcuvs
// #include <cuda_runtime_api.h>
// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"

type Resource struct {
	resource C.cuvsResources_t
}

// func NewResource() *Resource {
func NewResource(stream C.cudaStream_t) Resource {

	res := C.cuvsResources_t(0)

	CheckCuvs(C.cuvsResourcesCreate(&res))

	if stream != nil {
		CheckCuvs(C.cuvsStreamSet(res, stream))
	}

	return Resource{resource: res}
}

func Sync(r C.cuvsResources_t) {
	CheckCuvs(C.cuvsStreamSync(r))
}
