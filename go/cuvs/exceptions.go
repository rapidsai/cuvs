package common

// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"

func CheckCuvs(error C.cuvsError_t) {
	if error == C.CUVS_ERROR {
		panic(C.GoString(C.cuvsGetLastErrorText()))
	}
}

func CheckCuda(error C.cudaError_t) {
	if error != C.cudaSuccess {
		panic(C.GoString(C.cudaGetErrorString(error)))
	}
}
