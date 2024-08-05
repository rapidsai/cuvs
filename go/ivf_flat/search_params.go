package ivf_flat

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

type searchParams struct {
	params C.cuvsIvfFlatSearchParams_t
}

func CreateSearchParams() (*searchParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsIvfFlatSearchParams{})

	params := (C.cuvsIvfFlatSearchParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

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
	// TODO free memory
	return nil
}
