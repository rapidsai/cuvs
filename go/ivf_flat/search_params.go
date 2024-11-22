package ivf_flat

// #include <cuvs/neighbors/ivf_flat.h>
import "C"

import (
	cuvs "github.com/rapidsai/cuvs/go"
)

type searchParams struct {
	params C.cuvsIvfFlatSearchParams_t
}

func CreateSearchParams() (*searchParams, error) {
	var params C.cuvsIvfFlatSearchParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatSearchParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &searchParams{params: params}, nil
}

func (p *searchParams) SetNProbes(n_probes uint32) (*searchParams, error) {
	p.params.n_probes = C.uint32_t(n_probes)
	return p, nil
}

func (p *searchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
