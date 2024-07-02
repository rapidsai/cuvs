package ivf_pq

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

type IndexParams struct {
	params C.cuvsIvfFlatIndexParams_t
}

func CreateIndexParams(n_lists uint32, metric string, metric_arg float32, set_kmeans_n_iters uint32, kmeans_trainset_fraction float64, pq_bits uint32, pq_dim uint32, codebook_kind uint32, force_random_rotation bool, add_data_on_build bool) (*IndexParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsIvfFlatIndexParams{})

	params := (C.cuvsIvfFlatIndexParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfFlatIndexParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	CMetric := C.cuvsDistanceType(0)

	switch metric {
	case "L2Expanded":
		CMetric = C.L2Expanded
	default:
		return nil, errors.New("unsupported metric")
	}

	params.n_lists = C.uint32_t(n_lists)
	params.metric = C.cuvsDistanceType(CMetric)
	params.metric_arg = C.float(metric_arg)
	params.kmeans_n_iters = C.uint32_t(set_kmeans_n_iters)
	params.kmeans_trainset_fraction = C.double(kmeans_trainset_fraction)
	if add_data_on_build {
		params.add_data_on_build = C._Bool(true)
	} else {
		params.add_data_on_build = C._Bool(false)
	}

	return &IndexParams{params: params}, nil
}

func (p *IndexParams) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfFlatIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
