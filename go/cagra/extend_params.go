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

type ExtendParams struct {
	params C.cuvsCagraExtendParams_t
}

func CreateExtendParams() (*ExtendParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsCagraExtendParams{})

	params := (C.cuvsCagraExtendParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtendParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	ExtendParams := &ExtendParams{params: params}

	return ExtendParams, nil
}

func (p *ExtendParams) SetMaxChunkSize(max_chunk_size uint32) (*ExtendParams, error) {
	p.params.max_chunk_size = C.uint32_t(max_chunk_size)
	return p, nil
}

func (p *ExtendParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtendParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
