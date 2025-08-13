package cuvs

// #include <cuvs/core/c_api.h>
import "C"
import "errors"

type CuvsError C.cuvsError_t

// Wrapper function to convert cuvs error to Go error
func CheckCuvs(error CuvsError) error {
	if error == C.CUVS_ERROR {
		return errors.New(C.GoString(C.cuvsGetLastErrorText()))
	}
	return nil
}

// Wrapper function to convert cuda error to Go error
func CheckCuda(error C.cudaError_t) error {
	if error != C.cudaSuccess {
		return errors.New(C.GoString(C.cudaGetErrorString(error)))
	}
	return nil
}
