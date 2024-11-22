package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	cuvs "github.com/rapidsai/cuvs/go"
)

type ExtendParams struct {
	params C.cuvsCagraExtendParams_t
}

func CreateExtendParams() (*ExtendParams, error) {
	var params C.cuvsCagraExtendParams_t

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
	return nil
}
