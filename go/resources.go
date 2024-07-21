package cuvs

// #include <cuda_runtime_api.h>
// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"

type cuvsResource C.cuvsResources_t

type Resource struct {
	Resource C.cuvsResources_t
}

// func NewResource() *Resource {
func NewResource(stream C.cudaStream_t) (Resource, error) {

	res := C.cuvsResources_t(0)

	err := CheckCuvs(CuvsError(C.cuvsResourcesCreate(&res)))
	if err != nil {
		return Resource{}, err
	}

	if stream != nil {
		err := CheckCuvs(CuvsError(C.cuvsStreamSet(res, stream)))
		if err != nil {
			return Resource{}, err
		}
	}

	return Resource{Resource: res}, nil
}

func (r Resource) Sync() error {
	return CheckCuvs(CuvsError(C.cuvsStreamSync(r.Resource)))
}

func (r Resource) GetCudaStream() (C.cudaStream_t, error) {
	var stream C.cudaStream_t
	err := CheckCuvs(CuvsError(C.cuvsStreamGet(r.Resource, &stream)))
	if err != nil {
		return C.cudaStream_t(nil), err
	}
	return stream, nil
}

func (r Resource) Close() error {
	err := CheckCuvs(CuvsError(C.cuvsResourcesDestroy(r.Resource)))
	if err != nil {
		return err
	}

	return nil
}
