/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::convert::From;

use crate::error::{check_cuda, Result};
use crate::resources::Resources;

#[derive(Debug)]
pub struct ManagedTensor(ffi::DLManagedTensor);

pub trait IntoDtype {
    fn ffi_dtype() -> ffi::DLDataType;
}

impl ManagedTensor {
    pub fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        &self.0 as *const _ as *mut _
    }

    fn bytes(&self) -> usize {
        // figure out how many bytes to allocate
        let mut bytes: usize = 1;
        for x in 0..self.0.dl_tensor.ndim {
            bytes *= unsafe { (*self.0.dl_tensor.shape.add(x as usize)) as usize };
        }
        bytes *= (self.0.dl_tensor.dtype.bits / 8) as usize;
        bytes
    }

    pub fn to_device(&self, _res: &Resources) -> Result<ManagedTensor> {
        unsafe {
            let bytes = self.bytes();
            let mut device_data: *mut std::ffi::c_void = std::ptr::null_mut();

            // allocate storage, copy over
            check_cuda(ffi::cudaMalloc(&mut device_data as *mut _, bytes))?;
            check_cuda(ffi::cudaMemcpy(
                device_data,
                self.0.dl_tensor.data,
                bytes,
                ffi::cudaMemcpyKind_cudaMemcpyDefault,
            ))?;

            let mut ret = self.0.clone();
            ret.dl_tensor.data = device_data;
            // call cudaFree automatically to clean up data
            ret.deleter = Some(cuda_free_tensor);
            ret.dl_tensor.device.device_type = ffi::DLDeviceType::kDLCUDA;

            Ok(ManagedTensor(ret))
        }
    }
    pub fn to_host<
        T: IntoDtype,
        S: ndarray::RawData<Elem = T> + ndarray::RawDataMut,
        D: ndarray::Dimension,
    >(
        &self,
        _res: &Resources,
        arr: &mut ndarray::ArrayBase<S, D>,
    ) -> Result<()> {
        unsafe {
            let bytes = self.bytes();
            check_cuda(ffi::cudaMemcpy(
                arr.as_mut_ptr() as *mut std::ffi::c_void,
                self.0.dl_tensor.data,
                bytes,
                ffi::cudaMemcpyKind_cudaMemcpyDefault,
            ))?;

            Ok(())
        }
    }
}

unsafe extern "C" fn cuda_free_tensor(self_: *mut ffi::DLManagedTensor) {
    let _ = ffi::cudaFree((*self_).dl_tensor.data);
}

/// Create a non-owning view of a Tensor from a ndarray
impl<T: IntoDtype, S: ndarray::RawData<Elem = T>, D: ndarray::Dimension>
    From<&ndarray::ArrayBase<S, D>> for ManagedTensor
{
    fn from(arr: &ndarray::ArrayBase<S, D>) -> Self {
        // There is a draft PR out right now for creating dlpack directly from ndarray
        // right now, but until its merged we have to implement ourselves
        //https://github.com/rust-ndarray/ndarray/pull/1306/files
        unsafe {
            let mut ret = std::mem::MaybeUninit::<ffi::DLTensor>::uninit();
            let tensor = ret.as_mut_ptr();
            (*tensor).data = arr.as_ptr() as *mut std::os::raw::c_void;
            (*tensor).device = ffi::DLDevice {
                device_type: ffi::DLDeviceType::kDLCPU,
                device_id: 0,
            };
            (*tensor).byte_offset = 0;
            (*tensor).strides = std::ptr::null_mut(); // TODO: error if not rowmajor
            (*tensor).ndim = arr.ndim() as i32;
            (*tensor).shape = arr.shape().as_ptr() as *mut _;
            (*tensor).dtype = T::ffi_dtype();
            ManagedTensor(ffi::DLManagedTensor {
                dl_tensor: ret.assume_init(),
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            })
        }
    }
}

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.0.deleter {
                deleter(&mut self.0 as *mut _);
            }
        }
    }
}

impl IntoDtype for f32 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType {
            code: ffi::DLDataTypeCode::kDLFloat as _,
            bits: 32,
            lanes: 1,
        }
    }
}

impl IntoDtype for f64 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType {
            code: ffi::DLDataTypeCode::kDLFloat as _,
            bits: 64,
            lanes: 1,
        }
    }
}

impl IntoDtype for i32 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType {
            code: ffi::DLDataTypeCode::kDLInt as _,
            bits: 32,
            lanes: 1,
        }
    }
}

impl IntoDtype for u32 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType {
            code: ffi::DLDataTypeCode::kDLUInt as _,
            bits: 32,
            lanes: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ndarray() {
        let arr = ndarray::Array::<f32, _>::zeros((8, 4));

        let tensor = unsafe { (*(ManagedTensor::from(&arr).as_ptr())).dl_tensor };

        assert_eq!(tensor.ndim, 2);

        // make sure we can get the shape ok
        assert_eq!(unsafe { *tensor.shape }, 8);
        assert_eq!(unsafe { *tensor.shape.add(1) }, 4);
    }
}
