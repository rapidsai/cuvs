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

var cBuildAlgos = map[BuildAlgo]int{
	IvfPq:      C.IVF_PQ,
	NnDescent:  C.NN_DESCENT,
	AutoSelect: C.AUTO_SELECT,
}

func CreateCompressionParams() (*CompressionParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsCagraCompressionParams{})

	params := (C.cuvsCagraCompressionParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraCompressionParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	// params.pq_bits = C.uint32_t(pq_bits)
	// params.pq_dim = C.uint32_t(pq_dim)
	// params.vq_n_centers = C.uint32_t(vq_n_centers)
	// params.kmeans_n_iters = C.uint32_t(kmeans_n_iters)
	// params.vq_kmeans_trainset_fraction = C.double(vq_kmeans_trainset_fraction)
	// params.pq_kmeans_trainset_fraction = C.double(pq_kmeans_trainset_fraction)

	return &CompressionParams{params: params}, nil
}

func (p *CompressionParams) SetPQBits(pq_bits uint32) (*CompressionParams, error) {
	p.params.pq_bits = C.uint32_t(pq_bits)
	return p, nil
}

func (p *CompressionParams) SetPQDim(pq_dim uint32) (*CompressionParams, error) {
	p.params.pq_dim = C.uint32_t(pq_dim)
	return p, nil
}

func (p *CompressionParams) SetVQNCenters(vq_n_centers uint32) (*CompressionParams, error) {
	p.params.vq_n_centers = C.uint32_t(vq_n_centers)
	return p, nil
}

func (p *CompressionParams) SetKMeansNIters(kmeans_n_iters uint32) (*CompressionParams, error) {
	p.params.kmeans_n_iters = C.uint32_t(kmeans_n_iters)
	return p, nil
}

func (p *CompressionParams) SetVQKMeansTrainsetFraction(vq_kmeans_trainset_fraction float64) (*CompressionParams, error) {
	p.params.vq_kmeans_trainset_fraction = C.double(vq_kmeans_trainset_fraction)
	return p, nil
}

func (p *CompressionParams) SetPQKMeansTrainsetFraction(pq_kmeans_trainset_fraction float64) (*CompressionParams, error) {
	p.params.pq_kmeans_trainset_fraction = C.double(pq_kmeans_trainset_fraction)
	return p, nil
}

func CreateIndexParams() (*IndexParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsCagraIndexParams{})

	params := (C.cuvsCagraIndexParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	// CBuildAlgo, exists := CBuildAlgos[build_algo]

	// if !exists {
	// 	return nil, errors.New("cuvs: invalid build_algo")
	// }

	// params.intermediate_graph_degree = C.size_t(intermediate_graph_degree)
	// params.graph_degree = C.size_t(graph_degree)
	// params.build_algo = uint32(CBuildAlgo)
	// params.nn_descent_niter = C.ulong(nn_descent_niter)
	// params.compression = C.cuvsCagraCompressionParams_t(compression.params)

	IndexParams := &IndexParams{params: params}

	return IndexParams, nil
}

func (p *IndexParams) SetIntermediateGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error) {
	p.params.intermediate_graph_degree = C.size_t(intermediate_graph_degree)
	return p, nil
}

func (p *IndexParams) SetGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error) {

	p.params.graph_degree = C.size_t(intermediate_graph_degree)

	return p, nil

}

func (p *IndexParams) SetBuildAlgo(build_algo BuildAlgo) (*IndexParams, error) {
	CBuildAlgo, exists := cBuildAlgos[build_algo]

	if !exists {
		return nil, errors.New("cuvs: invalid build_algo")
	}
	p.params.build_algo = uint32(CBuildAlgo)

	return p, nil
}

func (p *IndexParams) SetNNDescentNiter(nn_descent_niter uint32) (*IndexParams, error) {
	p.params.nn_descent_niter = C.ulong(nn_descent_niter)
	return p, nil
}

func (p *IndexParams) SetCompression(compression *CompressionParams) (*IndexParams, error) {
	p.params.compression = C.cuvsCagraCompressionParams_t(compression.params)
	return p, nil
}

func (p *IndexParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
