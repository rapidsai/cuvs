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
	"rapidsai/cuvs/cuvs/common"
	"unsafe"
)

type IndexParams struct {
	params C.cuvsCagraIndexParams_t
}

type CompressionParams struct {
	params C.cuvsCagraCompressionParams_t
}

type BuildAlgo int

const (
	IvfPq BuildAlgo = iota
	NnDescent
	AutoSelect
)

var CBuildAlgos = map[BuildAlgo]int{
	IvfPq:      C.IVF_PQ,
	NnDescent:  C.NN_DESCENT,
	AutoSelect: C.AUTO_SELECT,
}

func CreateCompressionParams(pq_bits uint32, pq_dim uint32, vq_n_centers uint32, kmeans_n_iters uint32, vq_kmeans_trainset_fraction float64, pq_kmeans_trainset_fraction float64) (*CompressionParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsCagraCompressionParams{})

	params := (C.cuvsCagraCompressionParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := common.CheckCuvs(common.CuvsError(C.cuvsCagraCompressionParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	params.pq_bits = C.uint32_t(pq_bits)
	params.pq_dim = C.uint32_t(pq_dim)
	params.vq_n_centers = C.uint32_t(vq_n_centers)
	params.kmeans_n_iters = C.uint32_t(kmeans_n_iters)
	params.vq_kmeans_trainset_fraction = C.double(vq_kmeans_trainset_fraction)
	params.pq_kmeans_trainset_fraction = C.double(pq_kmeans_trainset_fraction)

	return &CompressionParams{params: params}, nil
}

func CreateIndexParams(intermediate_graph_degree uintptr, graph_degree uintptr, build_algo BuildAlgo, nn_descent_niter uint32, compression *CompressionParams) (*IndexParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsCagraIndexParams{})

	params := (C.cuvsCagraIndexParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := common.CheckCuvs(common.CuvsError(C.cuvsCagraIndexParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	CBuildAlgo, exists := CBuildAlgos[build_algo]

	if !exists {
		return nil, errors.New("cuvs: invalid build_algo")
	}

	params.intermediate_graph_degree = C.size_t(intermediate_graph_degree)
	params.graph_degree = C.size_t(graph_degree)
	params.build_algo = uint32(CBuildAlgo)
	params.nn_descent_niter = C.ulong(nn_descent_niter)
	params.compression = C.cuvsCagraCompressionParams_t(compression.params)

	return &IndexParams{params: params}, nil
}

func (p *IndexParams) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsCagraIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
