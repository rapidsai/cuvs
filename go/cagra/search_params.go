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
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

type SearchParams struct {
	params C.cuvsCagraSearchParams_t
}

type SearchAlgo int

const (
	SearchAlgoSingleCta SearchAlgo = iota
	SearchAlgoMultiCta
	SearchAlgoMultiKernel
	SearchAlgoAuto
)

type HashmapMode int

const (
	HashmapModeHash HashmapMode = iota
	HashmapModeSmall
	HashmapModeAuto
)

func CreateSearchParams() (*SearchParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsCagraSearchParams{})

	params := (C.cuvsCagraSearchParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

func (p *SearchParams) SetMaxQueries(max_queries uintptr) (*SearchParams, error) {
	p.params.max_queries = C.size_t(max_queries)
	return p, nil
}

func (p *SearchParams) SetItopkSize(itopk_size uintptr) (*SearchParams, error) {
	p.params.itopk_size = C.size_t(itopk_size)
	return p, nil
}

func (p *SearchParams) SetMaxIterations(max_iterations uintptr) (*SearchParams, error) {
	p.params.max_iterations = C.size_t(max_iterations)
	return p, nil
}

func (p *SearchParams) SetAlgo(algo SearchAlgo) (*SearchParams, error) {
	CAlgo := C.SINGLE_CTA

	switch algo {
	case SearchAlgoSingleCta:
		CAlgo = C.SINGLE_CTA
	case SearchAlgoMultiCta:
		CAlgo = C.MULTI_CTA
	case SearchAlgoMultiKernel:
		CAlgo = C.MULTI_KERNEL
	case SearchAlgoAuto:
		CAlgo = C.AUTO
	default:
		return nil, errors.New("unsupported algo")
	}

	p.params.algo = uint32(CAlgo)

	return p, nil
}

func (p *SearchParams) SetTeamSize(team_size uintptr) (*SearchParams, error) {
	p.params.team_size = C.size_t(team_size)
	return p, nil
}

func (p *SearchParams) SetMinIterations(min_iterations uintptr) (*SearchParams, error) {
	p.params.min_iterations = C.size_t(min_iterations)
	return p, nil
}

func (p *SearchParams) SetThreadBlockSize(thread_block_size uintptr) (*SearchParams, error) {
	p.params.thread_block_size = C.size_t(thread_block_size)
	return p, nil
}

func (p *SearchParams) SetHashmapMode(hashmap_mode HashmapMode) (*SearchParams, error) {
	CHashMode := C.AUTO_HASH

	switch hashmap_mode {
	case HashmapModeHash:
		CHashMode = C.HASH
	case HashmapModeSmall:
		CHashMode = C.SMALL
	case HashmapModeAuto:
		CHashMode = C.AUTO_HASH
	default:
		return nil, errors.New("unsupported hashmap_mode")
	}

	p.params.hashmap_mode = uint32(CHashMode)

	return p, nil
}

func (p *SearchParams) SetHashmapMinBitlen(hashmap_min_bitlen uintptr) (*SearchParams, error) {
	p.params.hashmap_min_bitlen = C.size_t(hashmap_min_bitlen)
	return p, nil
}

func (p *SearchParams) SetHashmapMaxFillRate(hashmap_max_fill_rate float32) (*SearchParams, error) {
	p.params.hashmap_max_fill_rate = C.float(hashmap_max_fill_rate)
	return p, nil
}

func (p *SearchParams) SetNumRandomSamplings(num_random_samplings uint32) (*SearchParams, error) {
	p.params.num_random_samplings = C.uint32_t(num_random_samplings)
	return p, nil
}

func (p *SearchParams) SetRandXorMask(rand_xor_mask uint64) (*SearchParams, error) {
	p.params.rand_xor_mask = C.uint64_t(rand_xor_mask)
	return p, nil
}

func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
