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
	"rapidsai/cuvs/cuvs/common"
	"unsafe"
)

type SearchParams struct {
	params C.cuvsIvfFlatSearchParams_t
}

func CreateSearchParams(n_probes uint32) (*SearchParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsIvfFlatSearchParams{})

	params := (C.cuvsIvfFlatSearchParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfFlatSearchParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

func (p *SearchParams) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfFlatSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
