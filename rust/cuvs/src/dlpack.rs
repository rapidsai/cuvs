/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::marker::PhantomData;

use crate::error::{Error, Result, check_cuvs};
use crate::resources::Resources;

/// ManagedTensor is a wrapper around a dlpack DLManagedTensor object.
/// This lets you pass matrices in device or host memory into cuvs.
#[derive(Debug)]
pub struct ManagedTensor<'a> {
    tensor: ffi::DLManagedTensor,
    shape: Box<[i64]>,
    _borrow: PhantomData<&'a ()>,
}

pub trait IntoDtype {
    fn ffi_dtype() -> ffi::DLDataType;
}

impl<'a> ManagedTensor<'a> {
    pub(crate) fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        &self.tensor as *const _ as *mut _
    }

    fn new(
        data: *mut std::ffi::c_void,
        device_type: ffi::DLDeviceType,
        dtype: ffi::DLDataType,
        shape: Box<[i64]>,
        deleter: Option<unsafe extern "C" fn(self_: *mut ffi::DLManagedTensor)>,
    ) -> Self {
        let dl_tensor = ffi::DLTensor {
            data,
            device: ffi::DLDevice { device_type, device_id: 0 },
            ndim: shape.len() as i32,
            dtype,
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };

        let mut ret = Self {
            tensor: ffi::DLManagedTensor { dl_tensor, manager_ctx: std::ptr::null_mut(), deleter },
            shape,
            _borrow: PhantomData,
        };
        ret.tensor.dl_tensor.shape = ret.shape.as_mut_ptr();
        ret
    }

    /// Create a non-owning view of a row-major ndarray.
    pub fn from_ndarray<T, S, D>(arr: &'a ndarray::ArrayBase<S, D>) -> Result<Self>
    where
        T: IntoDtype,
        S: ndarray::RawData<Elem = T>,
        D: ndarray::Dimension,
    {
        let shape = ndarray_shape(arr)?;
        validate_standard_layout(arr)?;
        Ok(Self::new(
            arr.as_ptr() as *mut std::ffi::c_void,
            ffi::DLDeviceType::kDLCPU,
            T::ffi_dtype(),
            shape,
            None,
        ))
    }

    /// Creates a new ManagedTensor on the current GPU device, and copies
    /// the data into it.
    pub fn to_device(&self, res: &Resources) -> Result<ManagedTensor<'static>> {
        unsafe {
            let bytes = dl_tensor_bytes(&self.tensor.dl_tensor);
            let mut device_data: *mut std::ffi::c_void = std::ptr::null_mut();

            // allocate storage, copy over
            check_cuvs(ffi::cuvsRMMAlloc(res.0, &mut device_data as *mut _, bytes))?;

            let ret = ManagedTensor::new(
                device_data,
                ffi::DLDeviceType::kDLCUDA,
                self.tensor.dl_tensor.dtype,
                self.shape.clone(),
                Some(rmm_free_tensor),
            );

            check_cuvs(ffi::cuvsMatrixCopy(res.0, self.as_ptr(), ret.as_ptr()))?;

            Ok(ret)
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
        validate_host_output(&self.tensor.dl_tensor, self.shape.as_ref(), arr)?;

        unsafe {
            let mut dst_shape = ndarray_shape(arr)?;
            let mut dst = ffi::DLManagedTensor {
                dl_tensor: ffi::DLTensor {
                    data: arr.as_mut_ptr() as *mut std::ffi::c_void,
                    device: ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCPU, device_id: 0 },
                    ndim: dst_shape.len() as i32,
                    dtype: T::ffi_dtype(),
                    shape: dst_shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };

            check_cuvs(ffi::cuvsMatrixCopy(res.0, self.as_ptr(), &mut dst))?;
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
    bytes *= ((tensor.dtype.bits as usize * tensor.dtype.lanes as usize).div_ceil(8)) as usize;
    bytes
}

fn ndarray_shape<S, D>(arr: &ndarray::ArrayBase<S, D>) -> Result<Box<[i64]>>
where
    S: ndarray::RawData,
    D: ndarray::Dimension,
{
    if arr.ndim() > i32::MAX as usize {
        return Err(Error::InvalidArgument(format!(
            "ndarray rank {} does not fit in i32",
            arr.ndim()
        )));
    }

    arr.shape()
        .iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| {
                Error::InvalidArgument(format!("ndarray dimension {dim} does not fit in i64"))
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

fn validate_standard_layout<S, D>(arr: &ndarray::ArrayBase<S, D>) -> Result<()>
where
    S: ndarray::RawData,
    D: ndarray::Dimension,
{
    if arr.is_standard_layout() {
        Ok(())
    } else {
        Err(Error::InvalidArgument("ndarray must be in standard row-major layout".to_string()))
    }
}

fn validate_dtype(actual: ffi::DLDataType, expected: ffi::DLDataType) -> Result<()> {
    if actual.code == expected.code
        && actual.bits == expected.bits
        && actual.lanes == expected.lanes
    {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "dtype mismatch: tensor has code={}, bits={}, lanes={} but ndarray has code={}, bits={}, lanes={}",
            actual.code, actual.bits, actual.lanes, expected.code, expected.bits, expected.lanes
        )))
    }
}

fn validate_host_output<T, S, D>(
    src: &ffi::DLTensor,
    src_shape: &[i64],
    arr: &ndarray::ArrayBase<S, D>,
) -> Result<()>
where
    T: IntoDtype,
    S: ndarray::RawData<Elem = T>,
    D: ndarray::Dimension,
{
    validate_standard_layout(arr)?;
    let dst_shape = ndarray_shape(arr)?;
    if src_shape != dst_shape.as_ref() {
        return Err(Error::InvalidArgument(format!(
            "shape mismatch: tensor has shape {:?} but ndarray has shape {:?}",
            src_shape,
            dst_shape.as_ref()
        )));
    }
    validate_dtype(src.dtype, T::ffi_dtype())
}

unsafe extern "C" fn rmm_free_tensor(self_: *mut ffi::DLManagedTensor) {
    unsafe {
        let bytes = dl_tensor_bytes(&(*self_).dl_tensor);
        let res = Resources::new().unwrap();
        let _ = ffi::cuvsRMMFree(res.0, (*self_).dl_tensor.data as *mut _, bytes);
    }
}

impl<'a, T: IntoDtype, S: ndarray::RawData<Elem = T>, D: ndarray::Dimension>
    TryFrom<&'a ndarray::ArrayBase<S, D>> for ManagedTensor<'a>
{
    type Error = Error;

    fn try_from(arr: &'a ndarray::ArrayBase<S, D>) -> Result<Self> {
        ManagedTensor::from_ndarray(arr)
    }
}

impl Drop for ManagedTensor<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.tensor.deleter {
                deleter(&mut self.tensor as *mut _);
            }
        }
    }
}

impl IntoDtype for f32 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType { code: ffi::DLDataTypeCode::kDLFloat as _, bits: 32, lanes: 1 }
    }
}

impl IntoDtype for f64 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType { code: ffi::DLDataTypeCode::kDLFloat as _, bits: 64, lanes: 1 }
    }
}

impl IntoDtype for i32 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType { code: ffi::DLDataTypeCode::kDLInt as _, bits: 32, lanes: 1 }
    }
}

impl IntoDtype for i64 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType { code: ffi::DLDataTypeCode::kDLInt as _, bits: 64, lanes: 1 }
    }
}

impl IntoDtype for u32 {
    fn ffi_dtype() -> ffi::DLDataType {
        ffi::DLDataType { code: ffi::DLDataTypeCode::kDLUInt as _, bits: 32, lanes: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ndarray() {
        let arr = ndarray::Array::<f32, _>::zeros((8, 4));

        let tensor = ManagedTensor::from_ndarray(&arr).unwrap();
        let tensor = unsafe { (*tensor.as_ptr()).dl_tensor };

        assert_eq!(tensor.ndim, 2);

        // make sure we can get the shape ok
        assert_eq!(unsafe { *tensor.shape }, 8);
        assert_eq!(unsafe { *tensor.shape.add(1) }, 4);

        let arr = ndarray::Array::<f32, _>::zeros((8,));
        let tensor = ManagedTensor::from_ndarray(&arr).unwrap();
        let tensor = unsafe { (*tensor.as_ptr()).dl_tensor };
        assert_eq!(tensor.ndim, 1);
    }

    #[test]
    fn test_from_ndarray_rejects_non_standard_layout() {
        let arr = ndarray::Array::<f32, _>::zeros((8, 4));
        let view = arr.slice(ndarray::s![.., ..;2]);

        let err = ManagedTensor::from_ndarray(&view).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn test_to_host_validation_rejects_shape_mismatch() {
        let src = ndarray::Array::<f32, _>::zeros((8, 4));
        let tensor = ManagedTensor::from_ndarray(&src).unwrap();
        let mut dst = ndarray::Array::<f32, _>::zeros((8, 3));

        let err = validate_host_output::<f32, _, _>(
            &tensor.tensor.dl_tensor,
            tensor.shape.as_ref(),
            &mut dst,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn test_to_host_validation_rejects_dtype_mismatch() {
        let src = ndarray::Array::<f32, _>::zeros((8, 4));
        let tensor = ManagedTensor::from_ndarray(&src).unwrap();
        let mut dst = ndarray::Array::<f64, _>::zeros((8, 4));

        let err = validate_host_output::<f64, _, _>(
            &tensor.tensor.dl_tensor,
            tensor.shape.as_ref(),
            &mut dst,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
