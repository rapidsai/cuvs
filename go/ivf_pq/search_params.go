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
	"rapidsai/cuvs"
	"unsafe"
)

type SearchParams struct {
	params C.cuvsIvfPqSearchParams_t
}

type LutDtype int

const (
	Lut_Uint8 LutDtype = iota
	Lut_Uint16
	Lut_Uint32
	Lut_Uint64
	Lut_Int8
	Lut_Int16
	Lut_Int32
	Lut_Int64
)

var CLutDtypes = map[LutDtype]int{
	Lut_Uint8:  C.CUDA_R_8U,
	Lut_Uint16: C.CUDA_R_16U,
	Lut_Uint32: C.CUDA_R_32U,
	Lut_Uint64: C.CUDA_R_64U,
	Lut_Int8:   C.CUDA_R_8I,
	Lut_Int16:  C.CUDA_R_16I,
	Lut_Int32:  C.CUDA_R_32I,
	Lut_Int64:  C.CUDA_R_64I,
}

type InternalDistanceDtype int

const (
	InternalDistance_Float32 InternalDistanceDtype = iota
	InternalDistance_Float64
)

var CInternalDistanceDtypes = map[InternalDistanceDtype]int{
	InternalDistance_Float32: C.CUDA_R_32F,
	InternalDistance_Float64: C.CUDA_R_64F,
}

func CreateSearchParams(n_probes uint32, lut_dtype LutDtype, internal_distance_dtype InternalDistanceDtype) (*SearchParams, error) {

	CLutDtype, exists := CLutDtypes[LutDtype(lut_dtype)]

	if !exists {
		return nil, errors.New("cuvs: invalid lut_dtype")
	}

	CInternalDistanceDtype, exists := CInternalDistanceDtypes[InternalDistanceDtype(internal_distance_dtype)]

	if !exists {
		return nil, errors.New("cuvs: invalid internal_distance_dtype")
	}

	size := unsafe.Sizeof(C.struct_cuvsIvfPqSearchParams{})

	params := (C.cuvsIvfPqSearchParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearchParamsCreate(&params)))

	params.n_probes = C.uint32_t(n_probes)
	params.lut_dtype = C.cudaDataType_t(CLutDtype)
	params.internal_distance_dtype = C.cudaDataType_t(CInternalDistanceDtype)

	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
