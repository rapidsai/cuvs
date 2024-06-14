package common

// #cgo CFLAGS: -I/usr/local/cuda/include -I/home/ajit/miniforge3/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/home/ajit/miniforge3/lib -lcudart -lcudart -lcuvs
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
