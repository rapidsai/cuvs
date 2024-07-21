package cagra

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
	"rapidsai/cuvs"
	"unsafe"
)

type SearchParams struct {
	params C.cuvsCagraSearchParams_t
}

func CreateSearchParams(max_queries uintptr, itopk_size uintptr, max_iterations uintptr, algo string, team_size uintptr, min_iterations uintptr, thread_block_size uintptr, hashmap_mode string, hashmap_min_bitlen uintptr, hashmap_max_fill_rate float32, num_random_samplings uint32, rand_xor_mask uint64) (*SearchParams, error) {

	CAlgo := C.SINGLE_CTA
	switch algo {
	case "single_cta":
		CAlgo = C.SINGLE_CTA
	case "multi_cta":
		CAlgo = C.MULTI_CTA
	case "multi_kernel":
		CAlgo = C.MULTI_KERNEL
	case "auto":
		CAlgo = C.AUTO
	default:
		return nil, errors.New("unsupported algo")
	}

	CHashMode := C.AUTO_HASH
	switch hashmap_mode {
	case "hash":
		CHashMode = C.HASH
	case "small":
		CHashMode = C.SMALL
	case "auto":
		CHashMode = C.AUTO_HASH
	default:
		return nil, errors.New("unsupported hashmap_mode")
	}

	size := unsafe.Sizeof(C.struct_cuvsCagraSearchParams{})

	params := (C.cuvsCagraSearchParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsCreate(&params)))

	params.max_queries = C.size_t(max_queries)
	params.itopk_size = C.size_t(itopk_size)
	params.max_iterations = C.size_t(max_iterations)
	params.algo = uint32(CAlgo)
	params.team_size = C.size_t(team_size)
	params.min_iterations = C.size_t(min_iterations)
	params.thread_block_size = C.size_t(thread_block_size)
	params.hashmap_mode = uint32(CHashMode)
	params.hashmap_min_bitlen = C.size_t(hashmap_min_bitlen)
	params.hashmap_max_fill_rate = C.float(hashmap_max_fill_rate)
	params.num_random_samplings = C.uint32_t(num_random_samplings)
	params.rand_xor_mask = C.uint64_t(rand_xor_mask)

	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
