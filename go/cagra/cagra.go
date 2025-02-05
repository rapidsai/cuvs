package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

// Cagra ANN Index
type CagraIndex struct {
	index   C.cuvsCagraIndex_t
	trained bool
}

// Creates a new empty Cagra Index
func CreateIndex() (*CagraIndex, error) {
	var index C.cuvsCagraIndex_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &CagraIndex{index: index}, nil
}

// Builds a new Index from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for building the index
// * `dataset` - A row-major Tensor on either the host or device to index
// * `index` - CagraIndex to build
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *CagraIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

// Extends the index with additional data
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for extending the index
// * `additional_dataset` - A row-major Tensor on the device to extend the index with
// * `return_dataset` - A row-major Tensor on the device that will receive the extended dataset
// * `index` - CagraIndex to extend
func ExtendIndex[T any](Resources cuvs.Resource, params *ExtendParams, additional_dataset *cuvs.Tensor[T], return_dataset *cuvs.Tensor[T], index *CagraIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before calling extend")
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtend(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(additional_dataset.C_tensor)), index.index, (*C.DLManagedTensor)(unsafe.Pointer(return_dataset.C_tensor)))))
	if err != nil {
		return err
	}
	return nil
}

// Destroys the Cagra Index
func (index *CagraIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexDestroy(index.index)))
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
// * `queries` - A tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
// * `allowList` - List of indices to allow in the search, if nil, no filtering is applied
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *CagraIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[uint32], distances *cuvs.Tensor[T], allowList []uint32) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	var filter C.cuvsFilter
	bitset := createBitset(allowList)
	allowListTensor, err := cuvs.NewVector[uint32](bitset)
	if err != nil {
		return err
	}
	defer allowListTensor.Close()
	_, err = allowListTensor.ToDevice(&Resources)
	if err != nil {
		return err
	}
	if allowList == nil {
		filter = C.cuvsFilter{
			_type: C.NO_FILTER,
			addr:  C.uintptr_t(0),
		}
	} else {
		filter = C.cuvsFilter{
			_type: C.BITSET,
			addr:  C.uintptr_t(uintptr(unsafe.Pointer(allowListTensor.C_tensor))),
		}
	}
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), filter)))
}

func createBitset(allowList []uint32) []uint32 {
	// Calculate size needed for the bitset array
	// Each uint32 handles 32 bits, so we divide the max ID by 32 (shift right by 5)
	maxID := uint32(0)
	for _, id := range allowList {
		if id > maxID {
			maxID = id
		}
	}
	size := (maxID >> 5) + 1 // Division by 32, add 1 to handle remainder
	bitset := make([]uint32, size)
	for _, id := range allowList {
		// Calculate which uint32 in our array (divide by 32)
		arrayIndex := id >> 5
		// Calculate bit position within that uint32 (mod 32)
		bitPosition := id & 31 // equivalent to id % 32
		// Set the bit
		bitset[arrayIndex] |= 1 << bitPosition
	}
	return bitset
}
