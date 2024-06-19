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
	"unsafe"
)

type ManagedTensor = C.DLManagedTensor

func NewManagedTensor[T any](from_cai bool, shape []int, data []T, use_int64 bool) *C.DLManagedTensor {

	if len(shape) < 2 {
		panic("shape must be atleast 2")
	}

	dlm := (*C.DLManagedTensor)(C.malloc(C.size_t(unsafe.Sizeof(C.DLManagedTensor{}))))

	if dlm == nil {
		panic("memory allocation failed")
	}

	var devicetype C.DLDeviceType

	if from_cai {
		devicetype = C.kDLCUDA
	} else {
		devicetype = C.kDLCPU
	}

	device := C.DLDevice{
		device_type: devicetype,
		device_id:   0,
	}

	var dtype C.DLDataType
	if use_int64 {
		dtype = C.DLDataType{
			bits:  C.uchar(64),
			lanes: C.ushort(1),
			code:  C.kDLInt,
		}
	} else {
		dtype = C.DLDataType{
			bits:  C.uchar(32),
			lanes: C.ushort(1),
			code:  C.kDLFloat,
		}
	}

	dlm.dl_tensor.data = unsafe.Pointer(&data[0])

	dlm.dl_tensor.device = device

	dlm.dl_tensor.dtype = dtype
	dlm.dl_tensor.ndim = C.int(len(shape))
	dlm.dl_tensor.shape = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	dlm.dl_tensor.strides = nil
	dlm.dl_tensor.byte_offset = 0

	dlm.manager_ctx = nil
	dlm.deleter = (*[0]byte)(C.delete_tensor)

	return dlm

}

func GetBytes(t *C.DLManagedTensor) int {
	bytes := 1

	for dim := 0; dim < int(t.dl_tensor.ndim); dim++ {
		offset := unsafe.Pointer(uintptr(unsafe.Pointer(t.dl_tensor.shape)) + uintptr(dim)*unsafe.Sizeof(*t.dl_tensor.shape))

		// Convert the pointer to the correct type and dereference it to get the value
		dimSize := *(*C.long)(offset)

		bytes *= int(dimSize)
	}
	bytes *= int(t.dl_tensor.dtype.bits / 8)

	return bytes
}

func ToDevice(t *C.DLManagedTensor, res *Resource) *C.DLManagedTensor {
	bytes := GetBytes(t)

	// device_data := &C.void{}

	var DeviceDataPointer unsafe.Pointer
	// var DeviceDataPointerPointer *unsafe.Pointer = &DeviceDataPointer
	// var deviceData *C.void = nil
	println("host data location:")
	println(t.dl_tensor.data)
	println("device data pointer:")
	println(DeviceDataPointer)
	println("host data location:")
	println(t.dl_tensor.data)
	// t.dl_tensor.data = unsafe.Pointer(uintptr(t.dl_tensor.data) + uintptr(1200))
	// println("new host data location:")
	// println(t.dl_tensor.data)

	// CheckCuvs(C.cuvsRMMAlloc(res.resource, &DeviceDataPointer, C.size_t(bytes)))
	CheckCuda(C.cudaMalloc(&DeviceDataPointer, C.size_t(bytes)))

	println("device data pointer:")
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

	CheckCuda(
		C.cudaMemcpy(
			DeviceDataPointer,
			// t.dl_tensor.data,
			unsafe.Pointer(&hostData[0]),
			// DeviceDataPointer,
			// C.size_t(bytes),
			C.size_t(bytes),
			C.cudaMemcpyHostToDevice,

			// res.get_cuda_stream(),
			// GetCudaStream(res.resource),
		))

	println("done")

	t.dl_tensor.data = DeviceDataPointer

	return t

}
