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

#[derive(Debug)]
pub struct ManagedTensor(ffi::DLManagedTensor);

pub trait IntoDtype {
    fn ffi_dtype() -> ffi::DLDataType;
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

impl ManagedTensor {
    /// Create a non-owning view of a Tensor from a ndarray
    pub fn from_ndarray<T: IntoDtype, S: ndarray::RawData<Elem = T>, D: ndarray::Dimension>(
        arr: ndarray::ArrayBase<S, D>,
    ) -> ManagedTensor {
        // There is a draft PR out right now for creating dlpack directly from ndarray
        // right now, but until its merged we have to implement ourselves
        //https://github.com/rust-ndarray/ndarray/pull/1306/files
        unsafe {
            let mut ret = core::mem::MaybeUninit::<ffi::DLTensor>::uninit();
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

    pub fn as_ptr(self) -> *mut ffi::DLManagedTensor {
        &self.0 as *const _ as *mut _
    }

    pub fn into_inner(self) -> ffi::DLManagedTensor {
        self.0
    }
}

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        // TODO: if we have a deletr here, call it to free up the memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ndarray() {
        let arr = ndarray::Array::<f32, _>::zeros((8, 4));

        let tensor = ManagedTensor::from_ndarray(arr).into_inner().dl_tensor;

        assert_eq!(tensor.ndim, 2);

        // make sure we can get the shape ok
        assert_eq!(unsafe { *tensor.shape }, 8);
        assert_eq!(unsafe { *tensor.shape.add(1) }, 4);
    }
}
