package cuvs

// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"
import "errors"

type CuvsError C.cuvsError_t

func CheckCuvs(error CuvsError) error {
	if error == C.CUVS_ERROR {
		return errors.New(C.GoString(C.cuvsGetLastErrorText()))
	}
	return nil
}

func CheckCuda(error C.cudaError_t) error {
	if error != C.cudaSuccess {
		return errors.New(C.GoString(C.cudaGetErrorString(error)))
	}
	return nil
}
