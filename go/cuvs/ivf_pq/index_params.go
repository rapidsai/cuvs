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

func CreateIndexParams(n_lists uint32, metric string, metric_arg float32, kmeans_n_iters uint32, kmeans_trainset_fraction float64, pq_bits uint32, pq_dim uint32, codebook_kind string, force_random_rotation bool, add_data_on_build bool) (*IndexParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsIvfPqIndexParams{})

	params := (C.cuvsIvfPqIndexParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfPqIndexParamsCreate(&params)))

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

	CCodebookKind := C.codebook_gen(0)
	switch codebook_kind {
	case "subspace":
		CCodebookKind = C.PER_SUBSPACE
	case "cluster":
		CCodebookKind = C.PER_CLUSTER
	default:
		return nil, errors.New("unsupported codebook_kind")
	}

	params.n_lists = C.uint32_t(n_lists)
	params.metric = C.cuvsDistanceType(CMetric)
	params.metric_arg = C.float(metric_arg)
	params.kmeans_n_iters = C.uint32_t(kmeans_n_iters)
	params.kmeans_trainset_fraction = C.double(kmeans_trainset_fraction)
	params.pq_bits = C.uint32_t(pq_bits)
	params.pq_dim = C.uint32_t(pq_dim)
	params.codebook_kind = C.codebook_gen(CCodebookKind)
	if add_data_on_build {
		params.add_data_on_build = C._Bool(true)
	} else {
		params.add_data_on_build = C._Bool(false)
	}
	if force_random_rotation {
		params.force_random_rotation = C._Bool(true)
	} else {
		params.force_random_rotation = C._Bool(false)
	}
	if add_data_on_build {
		params.add_data_on_build = C._Bool(true)
	} else {
		params.add_data_on_build = C._Bool(false)
	}

	return &IndexParams{params: params}, nil
}

func (p *IndexParams) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfPqIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
