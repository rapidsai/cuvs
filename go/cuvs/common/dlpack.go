package common

// #include <stdio.h>
// #include <stdlib.h>
// #include <dlpack/dlpack.h>  // Replace with the actual header file containing DLManagedTensor
// void delete_tensor(DLManagedTensor *tensor){
//     free(tensor->dl_tensor.shape);
//     tensor->manager_ctx = NULL;
//     free(tensor);
// }
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
)

type ManagedTensor = *C.DLManagedTensor

type Tensor[T any] struct {
	C_tensor ManagedTensor
}

// func NewTensor[T any](from_cai bool, shape []int, data []T, use_int64 bool) (Tensor, error) {
func NewTensor[T any](from_cai bool, data [][]T) (Tensor[T], error) {

	shape := make([]int, 2)
	shape[0] = len(data)
	shape[1] = len(data[0])

	data_flat := make([]T, len(data)*len(data[0]))
	for i := range data {
		for j := range data[i] {
			data_flat[i*len(data[0])+j] = data[i][j]
		}
	}

	if len(shape) < 2 {
		return Tensor[T]{}, errors.New("shape must be atleast 2")
	}

	dlm := (*C.DLManagedTensor)(C.malloc(C.size_t(unsafe.Sizeof(C.DLManagedTensor{}))))

	if dlm == nil {
		return Tensor[T]{}, errors.New("memory allocation failed")
	}

	device := C.DLDevice{
		device_type: C.DLDeviceType(C.kDLCPU),
		device_id:   0,
	}

	var dtype C.DLDataType
	switch any(data[0][0]).(type) {
	case int64:
		dtype = C.DLDataType{
			bits:  C.uchar(64),
			lanes: C.ushort(1),
			code:  C.kDLInt,
		}
	case float32:
		dtype = C.DLDataType{
			bits:  C.uchar(32),
			lanes: C.ushort(1),
			code:  C.kDLFloat,
		}
	default:
		return Tensor[T]{}, errors.New("unsupported data type")
	}

	dlm.dl_tensor.data = unsafe.Pointer(&data_flat[0])

	dlm.dl_tensor.device = device

	dlm.dl_tensor.dtype = dtype
	dlm.dl_tensor.ndim = C.int(len(shape))
	dlm.dl_tensor.shape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	dlm.dl_tensor.strides = nil
	dlm.dl_tensor.byte_offset = 0

	dlm.manager_ctx = nil
	dlm.deleter = nil

	return Tensor[T]{c_tensor: dlm}, nil

}

func (t *Tensor[T]) GetBytes() int {
	bytes := 1

	for dim := 0; dim < int(t.c_tensor.dl_tensor.ndim); dim++ {
		offset := unsafe.Pointer(uintptr(unsafe.Pointer(t.c_tensor.dl_tensor.shape)) + uintptr(dim)*unsafe.Sizeof(*t.c_tensor.dl_tensor.shape))

		// Convert the pointer to the correct type and dereference it to get the value
		dimSize := *(*C.long)(offset)

		bytes *= int(dimSize)
	}
	bytes *= int(t.c_tensor.dl_tensor.dtype.bits / 8)

	return bytes
}

func (t *Tensor[T]) Close() error {
	// TODO: free memory
	if t.c_tensor.dl_tensor.device.device_type == C.kDLCUDA {
		bytes := t.GetBytes()
		res, err := NewResource(nil)
		if err != nil {
			return err
		}
		return CheckCuvs(C.cuvsRMMFree(res.Resource, t.c_tensor.dl_tensor.data, C.size_t(bytes)))

		// C.run_callback(t.c_tensor.deleter, t.c_tensor)
	}
	return nil

}

func (t *Tensor[T]) ToDevice(res *Resource) (*Tensor[T], error) {
	bytes := t.GetBytes()

	// device_data := &C.void{}

	var DeviceDataPointer unsafe.Pointer
	// var DeviceDataPointerPointer *unsafe.Pointer = &DeviceDataPointer
	// var deviceData *C.void = nil
	println("host data location:")
	println(t.c_tensor.dl_tensor.data)
	println("device data pointer:")
	println(DeviceDataPointer)
	println("host data location:")
	println(t.c_tensor.dl_tensor.data)

	err := CheckCuvs(C.cuvsRMMAlloc(res.Resource, &DeviceDataPointer, C.size_t(bytes)))
	if err != nil {
		//	panic(err)
		return nil, err
	}
	// CheckCuda(C.cudaMalloc(&DeviceDataPointer, C.size_t(bytes)))

	println("device data pointer (after allocation):")
	println(DeviceDataPointer)
	println(&DeviceDataPointer)
	println("bytes:")
	println(bytes)
	// bytes = 0

	hostData := make([]float32, 2)

	// Initialize host memory in Go
	for i := range hostData {
		hostData[i] = float32(i)
	}

	println("host data:")
	println(unsafe.Pointer(&hostData[0]))

	err = CheckCuda(
		C.cudaMemcpy(
			DeviceDataPointer,
			t.c_tensor.dl_tensor.data,
			C.size_t(bytes),
			C.cudaMemcpyHostToDevice,
		))

	if err != nil {
		return nil, err
	}
	t.c_tensor.dl_tensor.device.device_type = C.kDLCUDA
	t.c_tensor.dl_tensor.data = DeviceDataPointer
	println("normal transfer done")

	return t, nil

}

func (t *Tensor[T]) ToHost(res *Resource) (*Tensor[T], error) {
	bytes := t.GetBytes()

	addr := (C.malloc(C.size_t(bytes)))

	err := CheckCuda(
		C.cudaMemcpy(

			addr,

			t.c_tensor.dl_tensor.data,

			C.size_t(bytes),
			C.cudaMemcpyDeviceToHost,
		))

	if err != nil {
		return nil, err
	}

	t.c_tensor.dl_tensor.device.device_type = C.kDLCPU
	t.c_tensor.dl_tensor.data = addr

	return t, nil
}

func (t *Tensor[T]) GetArray() ([][]T, error) {
	if t.c_tensor.dl_tensor.device.device_type != C.kDLCPU {
		return nil, errors.New("Tensor must be on CPU")
	}

	shape := unsafe.Slice((*int64)(t.c_tensor.dl_tensor.shape), 2)

	data_flat := unsafe.Slice((*T)(t.c_tensor.dl_tensor.data), shape[0]*shape[1])

	data := make([][]T, shape[0])
	for i := range data {
		data[i] = make([]T, shape[1])
		for j := range data[i] {
			data[i][j] = data_flat[i*int(shape[1])+j]
		}
	}

	return data, nil

}

func TestCuda() {

	var DeviceDataPointer unsafe.Pointer
	CheckCuda(C.cudaMalloc(&DeviceDataPointer, C.size_t(8)))

	array := make([]float32, 2)

	for i := range array {
		array[i] = float32(i)
	}

	CheckCuda(
		C.cudaMemcpy(
			// DeviceDataPointer,
			DeviceDataPointer,
			unsafe.Pointer(&array[0]),
			// unsafe.Pointer(&hostData[0]),
			// DeviceDataPointer,
			// C.size_t(bytes),
			C.size_t(8),
			C.cudaMemcpyHostToDevice,

			// res.get_cuda_stream(),
			// GetCudaStream(res.resource),
		))

	println("cuda test done")

}
