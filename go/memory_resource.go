package cuvs

// #include <cuda_runtime_api.h>
// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"

func EnablePoolMemoryResource(initial_pool_size_percent int, max_pool_size_percent int) error {
	return CheckCuvs(CuvsError(C.cuvsRMMPoolMemoryResourceEnable(C.int(initial_pool_size_percent), C.int(max_pool_size_percent))))
}

func ResetMemoryResource() error {
	return CheckCuvs(CuvsError(C.cuvsRMMMemoryResourceReset()))
}
