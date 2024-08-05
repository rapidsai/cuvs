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

	cuvs "github.com/rapidsai/cuvs/go"

	"unsafe"
)

type indexParams struct {
	params C.cuvsIvfPqIndexParams_t
}

type codebookKind int

const (
	Subspace codebookKind = iota
	Cluster
)

var cCodebookKinds = map[codebookKind]int{
	Subspace: C.PER_SUBSPACE,
	Cluster:  C.PER_CLUSTER,
}

func CreateIndexParams() (*indexParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsIvfPqIndexParams{})

	params := (C.cuvsIvfPqIndexParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexParamsCreate(&params)))

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

func (p *indexParams) SetPQBits(pq_bits uint32) (*indexParams, error) {
	p.params.pq_bits = C.uint32_t(pq_bits)
	return p, nil
}

func (p *indexParams) SetPQDim(pq_dim uint32) (*indexParams, error) {
	p.params.pq_dim = C.uint32_t(pq_dim)
	return p, nil
}

func (p *indexParams) SetCodebookKind(codebook_kind codebookKind) (*indexParams, error) {
	CCodebookKind, exists := cCodebookKinds[codebook_kind]

	if !exists {
		return nil, errors.New("cuvs: invalid codebook_kind")
	}
	p.params.codebook_kind = uint32(CCodebookKind)

	return p, nil
}

func (p *indexParams) SetForceRandomRotation(force_random_rotation bool) (*indexParams, error) {
	if force_random_rotation {
		p.params.force_random_rotation = C._Bool(true)
	} else {
		p.params.force_random_rotation = C._Bool(false)
	}
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
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
