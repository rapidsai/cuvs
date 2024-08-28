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
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

type searchParams struct {
	params C.cuvsIvfPqSearchParams_t
}

type lutDtype int

const (
	Lut_Uint8 lutDtype = iota
	Lut_Uint16
	Lut_Uint32
	Lut_Uint64
	Lut_Int8
	Lut_Int16
	Lut_Int32
	Lut_Int64
)

var cLutDtypes = map[lutDtype]int{
	Lut_Uint8:  C.CUDA_R_8U,
	Lut_Uint16: C.CUDA_R_16U,
	Lut_Uint32: C.CUDA_R_32U,
	Lut_Uint64: C.CUDA_R_64U,
	Lut_Int8:   C.CUDA_R_8I,
	Lut_Int16:  C.CUDA_R_16I,
	Lut_Int32:  C.CUDA_R_32I,
	Lut_Int64:  C.CUDA_R_64I,
}

type internalDistanceDtype int

const (
	InternalDistance_Float32 internalDistanceDtype = iota
	InternalDistance_Float64
)

var CInternalDistanceDtypes = map[internalDistanceDtype]int{
	InternalDistance_Float32: C.CUDA_R_32F,
	InternalDistance_Float64: C.CUDA_R_64F,
}

func CreateSearchParams() (*searchParams, error) {

	size := unsafe.Sizeof(C.struct_cuvsIvfPqSearchParams{})

	params := (C.cuvsIvfPqSearchParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearchParamsCreate(&params)))

	if err != nil {
		return nil, err
	}

	return &searchParams{params: params}, nil
}

func (p *searchParams) SetNProbes(n_probes uint32) (*searchParams, error) {
	p.params.n_probes = C.uint32_t(n_probes)
	return p, nil
}

func (p *searchParams) SetLutDtype(lut_dtype lutDtype) (*searchParams, error) {
	CLutDtype, exists := cLutDtypes[lutDtype(lut_dtype)]

	if !exists {
		return nil, errors.New("cuvs: invalid lut_dtype")
	}
	p.params.lut_dtype = C.cudaDataType_t(CLutDtype)

	return p, nil
}

func (p *searchParams) SetInternalDistanceDtype(internal_distance_dtype internalDistanceDtype) (*searchParams, error) {
	CInternalDistanceDtype, exists := CInternalDistanceDtypes[internalDistanceDtype(internal_distance_dtype)]

	if !exists {
		return nil, errors.New("cuvs: invalid internal_distance_dtype")
	}
	p.params.internal_distance_dtype = C.cudaDataType_t(CInternalDistanceDtype)

	return p, nil
}

func (p *searchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
