package common

// #include <stdio.h>
// #include <stdlib.h>
// #include <dlpack/dlpack.h>  // Replace with the actual header file containing DLManagedTensor
// void delete_tensor(DLManagedTensor *tensor){
//     free(tensor->dl_tensor.shape);
//     tensor->manager_ctx = NULL;
//     free(tensor);
// }
import "C"
import (
	"unsafe"
)

type ManagedTensor = C.DLManagedTensor

func NewManagedTensor(from_cai bool, shape []int, data []float32) *C.DLManagedTensor {

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

	dtype := C.DLDataType{
		bits:  C.uchar(32),
		lanes: C.ushort(1),
		code:  C.kDLFloat,
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
