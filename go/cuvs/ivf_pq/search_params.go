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

type SearchParams struct {
	params C.cuvsIvfFlatSearchParams_t
}

func CreateSearchParams(n_probes uint32, lut_dtype string, internal_distance_dtype string) (*SearchParams, error) {

	CLutDtype := C.cudaDataType_t(0)
	switch lut_dtype {
	case "uint8":
		CLutDtype = C.CUDA_R_8U
	case "uint16":
		CLutDtype = C.CUDA_R_16U
	case "uint32":
		CLutDtype = C.CUDA_R_32U
	case "uint64":
		CLutDtype = C.CUDA_R_64U
	case "int8":
		CLutDtype = C.CUDA_R_8I
	case "int16":
		CLutDtype = C.CUDA_R_16I
	case "int32":
		CLutDtype = C.CUDA_R_32I
	case "int64":
		CLutDtype = C.CUDA_R_64I
	default:
		return nil, errors.New("unsupported lut_dtype")
	}

	CInternalDistanceDtype := C.cudaDataType_t(0)
	switch internal_distance_dtype {
	case "float32":
		CInternalDistanceDtype = C.CUDA_R_32F
	case "float64":
		CInternalDistanceDtype = C.CUDA_R_64F
	default:
		return nil, errors.New("unsupported internal_distance_dtype")
	}

	size := unsafe.Sizeof(C.struct_cuvsIvfPqSearchParams{})

	params := (C.cuvsIvfPqSearchParams_t)(C.malloc(C.size_t(size)))

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfPqSearchParamsCreate(&params)))

	params.n_probes = C.uint32_t(n_probes)
	params.lut_dtype = C.cudaDataType_t(CLutDtype)
	params.internal_distance_dtype = C.cudaDataType_t(CInternalDistanceDtype)

	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

func (p *SearchParams) Close() error {
	err := common.CheckCuvs(common.CuvsError(C.cuvsIvfFlatSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	// TODO free memory
	return nil
}
