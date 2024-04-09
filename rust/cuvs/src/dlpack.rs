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

use crate::error::{check_cuda, check_cuvs, Result};
use crate::resources::Resources;

/// ManagedTensor is a wrapper around a dlpack DLManagedTensor object.
/// This lets you pass matrices in device or host memory into cuvs.
#[derive(Debug)]
pub struct ManagedTensor(ffi::DLManagedTensor);

pub trait IntoDtype {
    fn ffi_dtype() -> ffi::DLDataType;
}

impl ManagedTensor {
    pub fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        &self.0 as *const _ as *mut _
    }

    /// Creates a new ManagedTensor on the current GPU device, and copies
    /// the data into it.
    pub fn to_device(&self, res: &Resources) -> Result<ManagedTensor> {
        unsafe {
            let bytes = dl_tensor_bytes(&self.0.dl_tensor);
            let mut device_data: *mut std::ffi::c_void = std::ptr::null_mut();

            // allocate storage, copy over
            check_cuvs(ffi::cuvsRMMAlloc(res.0, &mut device_data as *mut _, bytes))?;

            check_cuda(ffi::cudaMemcpyAsync(
                device_data,
                self.0.dl_tensor.data,
                bytes,
                ffi::cudaMemcpyKind_cudaMemcpyDefault,
                res.get_cuda_stream()?,
            ))?;

            let mut ret = self.0.clone();
            ret.dl_tensor.data = device_data;
            ret.deleter = Some(rmm_free_tensor);
            ret.dl_tensor.device.device_type = ffi::DLDeviceType::kDLCUDA;

            Ok(ManagedTensor(ret))
        }
    }

    /// Copies data from device memory into host memory
    pub fn to_host<
        T: IntoDtype,
        S: ndarray::RawData<Elem = T> + ndarray::RawDataMut,
        D: ndarray::Dimension,
    >(
        &self,
        res: &Resources,
        arr: &mut ndarray::ArrayBase<S, D>,
    ) -> Result<()> {
        unsafe {
            let bytes = dl_tensor_bytes(&self.0.dl_tensor);
            check_cuda(ffi::cudaMemcpyAsync(
                arr.as_mut_ptr() as *mut std::ffi::c_void,
                self.0.dl_tensor.data,
                bytes,
                ffi::cudaMemcpyKind_cudaMemcpyDefault,
                res.get_cuda_stream()?,
            ))?;
            Ok(())
        }
    }
}

/// Figures out how many bytes are in a DLTensor
fn dl_tensor_bytes(tensor: &ffi::DLTensor) -> usize {
    let mut bytes: usize = 1;
    for dim in 0..tensor.ndim {
        bytes *= unsafe { (*tensor.shape.add(dim as usize)) as usize };
    }
    bytes *= (tensor.dtype.bits / 8) as usize;
    bytes
}

unsafe extern "C" fn rmm_free_tensor(self_: *mut ffi::DLManagedTensor) {
    let bytes = dl_tensor_bytes(&(*self_).dl_tensor);
    let res = Resources::new().unwrap();
    let _ = ffi::cuvsRMMFree(res.0, (*self_).dl_tensor.data as *mut _, bytes);
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

impl IntoDtype for i64 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType {
            code: ffi::DLDataTypeCode::kDLInt as _,
            bits: 64,
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
