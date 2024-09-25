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

type indexParams struct {
	params C.cuvsIvfFlatIndexParams_t
}

func CreateIndexParams() (*indexParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsIvfFlatIndexParams{})

	params := (C.cuvsIvfFlatIndexParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatIndexParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	return &indexParams{params: params}, nil
}

func (p *indexParams) SetNLists(n_lists uint32) (*indexParams, error) {
	p.params.n_lists = C.uint32_t(n_lists)
	return p, nil
}

func (p *indexParams) SetMetric(metric cuvs.Distance) (*indexParams, error) {
	CMetric, exists := cuvs.CDistances[metric]

	if !exists {
		return nil, errors.New("cuvs: invalid distance metric")
	}
	p.params.metric = C.cuvsDistanceType(CMetric)

	return p, nil
}

func (p *indexParams) SetMetricArg(metric_arg float32) (*indexParams, error) {
	p.params.metric_arg = C.float(metric_arg)
	return p, nil
}

func (p *indexParams) SetKMeansNIters(kmeans_n_iters uint32) (*indexParams, error) {
	p.params.kmeans_n_iters = C.uint32_t(kmeans_n_iters)
	return p, nil
}

func (p *indexParams) SetKMeansTrainsetFraction(kmeans_trainset_fraction float64) (*indexParams, error) {
	p.params.kmeans_trainset_fraction = C.double(kmeans_trainset_fraction)
	return p, nil
}

func (p *indexParams) SetAddDataOnBuild(add_data_on_build bool) (*indexParams, error) {
	if add_data_on_build {
		p.params.add_data_on_build = C._Bool(true)
	} else {
		p.params.add_data_on_build = C._Bool(false)
	}
	return p, nil
}

func (p *indexParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
