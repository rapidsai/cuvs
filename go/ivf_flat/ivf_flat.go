package ivf_flat

// #include <cuvs/neighbors/ivf_flat.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

// IVF Flat Index
type IvfFlatIndex struct {
	index   C.cuvsIvfFlatIndex_t
	trained bool
}

// Creates a new empty IvfFlatIndex
func CreateIndex(params *IndexParams, dataset *cuvs.Tensor[float32]) (*IvfFlatIndex, error) {
	var index C.cuvsIvfFlatIndex_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &IvfFlatIndex{index: index}, nil
}

// Builds an IvfFlatIndex from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for building the index
// * `dataset` - A row-major Tensor on either the host or device to index
// * `index` - IvfFlatIndex to build
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *IvfFlatIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

// Destroys the IvfFlatIndex
func (index *IvfFlatIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Perform a Approximate Nearest Neighbors search on the Index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters to use in searching the index
// * `index` - IvfFlatIndex to search
// * `queries` - A tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *IvfFlatIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[T]) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}
        prefilter := C.cuvsFilter{
                addr:  0,
                _type: C.NO_FILTER,
        }

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), prefilter)))
}
