package cuvs

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
	// "github.com/rapidsai/cuvs/go"
	// "rapidsai/cuvs/ivf_flat"
	"unsafe"
)

type Tensor[T any] struct {
	C_tensor *C.DLManagedTensor
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
	case uint32:
		dtype = C.DLDataType{
			bits:  C.uchar(32),
			lanes: C.ushort(1),
			code:  C.kDLUInt,
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

	return Tensor[T]{C_tensor: dlm}, nil

}

func (t *Tensor[T]) GetBytes() int {
	bytes := 1

	for dim := 0; dim < int(t.C_tensor.dl_tensor.ndim); dim++ {
		offset := unsafe.Pointer(uintptr(unsafe.Pointer(t.C_tensor.dl_tensor.shape)) + uintptr(dim)*unsafe.Sizeof(*t.C_tensor.dl_tensor.shape))

		// Convert the pointer to the correct type and dereference it to get the value
		dimSize := *(*C.long)(offset)

		bytes *= int(dimSize)
	}
	bytes *= int(t.C_tensor.dl_tensor.dtype.bits / 8)

	return bytes
}

func (t *Tensor[T]) Close() error {
	// TODO: free memory
	if t.C_tensor.dl_tensor.device.device_type == C.kDLCUDA {
		bytes := t.GetBytes()
		res, err := NewResource(nil)
		if err != nil {
			return err
		}
		return CheckCuvs(CuvsError(C.cuvsRMMFree(res.Resource, t.C_tensor.dl_tensor.data, C.size_t(bytes))))

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
	// println("host data location:")
	// println(t.C_tensor.dl_tensor.data)
	// println("device data pointer:")
	// println(DeviceDataPointer)
	// println("host data location:")
	// println(t.C_tensor.dl_tensor.data)

	err := CheckCuvs(CuvsError(C.cuvsRMMAlloc(res.Resource, &DeviceDataPointer, C.size_t(bytes))))
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

	err = CheckCuda(
		C.cudaMemcpy(
			DeviceDataPointer,
			t.C_tensor.dl_tensor.data,
			C.size_t(bytes),
			C.cudaMemcpyHostToDevice,
		))

	if err != nil {
		return nil, err
	}
	t.C_tensor.dl_tensor.device.device_type = C.kDLCUDA
	t.C_tensor.dl_tensor.data = DeviceDataPointer
	println("normal transfer done")

	return t, nil

}

func (t *Tensor[T]) Expand(res *Resource, newData [][]T) (*Tensor[T], error) {

	if t.C_tensor.dl_tensor.device.device_type == C.kDLCPU {
		return &Tensor[T]{}, errors.New("Tensor must be on GPU")
	}

	new_shape := make([]int64, 2)
	new_shape[0] = int64(len(newData))
	new_shape[1] = int64(len(newData[0]))

	data_flat := make([]T, len(newData)*len(newData[0]))
	for i := range newData {
		for j := range newData[i] {
			data_flat[i*len(newData[0])+j] = newData[i][j]
		}
	}

	old_shape := unsafe.Slice((*int64)(t.C_tensor.dl_tensor.shape), 2)

	if old_shape[1] != new_shape[1] {
		return &Tensor[T]{}, errors.New("new shape must be same as old shape")
	}

	if len(new_shape) < 2 {
		return &Tensor[T]{}, errors.New("shape must be atleast 2")
	}

	newDataSize := 0

	switch any(newData[0][0]).(type) {
	case int64:
		// dtype = C.DLDataType{
		// 	bits:  C.uchar(64),
		// 	lanes: C.ushort(1),
		// 	code:  C.kDLInt,
		// }
		newDataSize = len(newData) * len(newData[0]) * 8
	case uint32:
		// dtype = C.DLDataType{
		// 	bits:  C.uchar(32),
		// 	lanes: C.ushort(1),
		// 	code:  C.kDLUInt,
		// }
		newDataSize = len(newData) * len(newData[0]) * 4
	case float32:
		// dtype = C.DLDataType{
		// 	bits:  C.uchar(32),
		// 	lanes: C.ushort(1),
		// 	code:  C.kDLFloat,
		// }
		newDataSize = len(newData) * len(newData[0]) * 4
	default:
		return &Tensor[T]{}, errors.New("unsupported data type")
	}

	bytes := t.GetBytes()

	var NewDeviceDataPointer unsafe.Pointer
	// var DeviceDataPointerPointer *unsafe.Pointer = &DeviceDataPointer
	// var deviceData *C.void = nil
	println("host data location:")
	println(t.C_tensor.dl_tensor.data)
	println("device data pointer:")
	println(NewDeviceDataPointer)
	println("host data location:")
	println(t.C_tensor.dl_tensor.data)

	err := CheckCuvs(CuvsError(C.cuvsRMMAlloc(res.Resource, &NewDeviceDataPointer, C.size_t(bytes+newDataSize))))
	if err != nil {
		//	panic(err)
		return nil, err
	}

	err = CheckCuda(
		C.cudaMemcpy(
			NewDeviceDataPointer,
			t.C_tensor.dl_tensor.data,
			C.size_t(bytes),
			C.cudaMemcpyDeviceToDevice,
		))

	if err != nil {
		return nil, err
	}

	err = CheckCuda(
		C.cudaMemcpy(
			unsafe.Pointer(uintptr(NewDeviceDataPointer)+uintptr(bytes)),
			unsafe.Pointer(&data_flat[0]),
			C.size_t(newDataSize),
			C.cudaMemcpyHostToDevice,
		))

	if err != nil {
		return nil, err
	}

	err = CheckCuvs(CuvsError(
		C.cuvsRMMFree(res.Resource, t.C_tensor.dl_tensor.data, C.size_t(bytes))))

	if err != nil {
		return nil, err
	}

	shape := make([]int64, 2)
	shape[0] = int64(*t.C_tensor.dl_tensor.shape) + int64(len(newData))
	println(old_shape[1])
	shape[1] = new_shape[1]

	t.C_tensor.dl_tensor.data = NewDeviceDataPointer
	t.C_tensor.dl_tensor.shape = (*C.int64_t)(unsafe.Pointer(&shape[0]))

	return t, nil
}

func (t *Tensor[T]) ToHost(res *Resource) (*Tensor[T], error) {
	bytes := t.GetBytes()

	addr := (C.malloc(C.size_t(bytes)))

	err := CheckCuda(
		C.cudaMemcpy(

			addr,

			t.C_tensor.dl_tensor.data,

			C.size_t(bytes),
			C.cudaMemcpyDeviceToHost,
		))

	if err != nil {
		return nil, err
	}

	err = CheckCuvs(CuvsError(
		C.cuvsRMMFree(res.Resource, t.C_tensor.dl_tensor.data, C.size_t(bytes))))

	if err != nil {
		return nil, err
	}

	t.C_tensor.dl_tensor.device.device_type = C.kDLCPU
	t.C_tensor.dl_tensor.data = addr

	return t, nil
}

func (t *Tensor[T]) GetArray() ([][]T, error) {
	if t.C_tensor.dl_tensor.device.device_type != C.kDLCPU {
		return nil, errors.New("Tensor must be on CPU")
	}

	shape := unsafe.Slice((*int64)(t.C_tensor.dl_tensor.shape), 2)

	data_flat := unsafe.Slice((*T)(t.C_tensor.dl_tensor.data), shape[0]*shape[1])

	data := make([][]T, shape[0])
	for i := range data {
		data[i] = make([]T, shape[1])
		for j := range data[i] {
			data[i][j] = data_flat[i*int(shape[1])+j]
		}
	}

	return data, nil

}
