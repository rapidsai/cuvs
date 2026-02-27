package cuvs

// #include <cuvs/core/c_api.h>
// #include <cuvs/version_config.h>
import "C"

type cuvsResource C.cuvsResources_t

// Resources are objects that are shared between function calls,
// and includes things like CUDA streams, cuBLAS handles and other
// resources that are expensive to create.
type Resource struct {
	Resource C.cuvsResources_t
}

type CudaStream struct {
	stream C.cudaStream_t
}

func (s *CudaStream) Close() error {
	err := CheckCuda(C.cudaStreamDestroy(s.stream))
	if err != nil {
		return err
	}
	s.stream = nil
	return nil
}

// Creates a new CUDA stream
func NewCudaStream() (*CudaStream, error) {
	var stream C.cudaStream_t
	err := CheckCuda(C.cudaStreamCreate(&stream))
	if err != nil {
		return nil, err
	}
	return &CudaStream{stream: stream}, nil
}

// Returns a new Resource object
func NewResource(stream *CudaStream) (Resource, error) {
	res := C.cuvsResources_t(0)
	err := CheckCuvs(CuvsError(C.cuvsResourcesCreate(&res)))
	if err != nil {
		return Resource{}, err
	}

	if stream != nil {
		err := CheckCuvs(CuvsError(C.cuvsStreamSet(res, stream.stream)))
		if err != nil {
			C.cuvsResourcesDestroy(res) // Clean up the resource created
			return Resource{}, err
		}
	}

	return Resource{Resource: res}, nil
}

// Syncs the current cuda stream
func (r Resource) Sync() error {
	return CheckCuvs(CuvsError(C.cuvsStreamSync(r.Resource)))
}

// Gets the current cuda stream
func (r Resource) GetCudaStream() (C.cudaStream_t, error) {
	var stream C.cudaStream_t

	err := CheckCuvs(CuvsError(C.cuvsStreamGet(r.Resource, &stream)))
	if err != nil {
		return C.cudaStream_t(nil), err
	}

	return stream, nil
}

func (r Resource) Close() error {
	err := CheckCuvs(CuvsError(C.cuvsResourcesDestroy(r.Resource)))
	if err != nil {
		return err
	}

	return nil
}
