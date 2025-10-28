#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.device_tensor_view import DeviceTensorView
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array

PQ_KMEANS_TYPES = {
    "kmeans" : cuvsKMeansType.KMeans,
    "kmeans_balanced" : cuvsKMeansType.KMeansBalanced}

PQ_KMEANS_NAMES = {v: k for k, v in PQ_KMEANS_TYPES.items()}

cdef class QuantizerParams:
    """
    Parameters for product quantization

    Parameters
    ----------
    pq_bits: int
        specifies the bit length of the vector element after compression by PQ
        possible values: [4, 5, 6, 7, 8]
    pq_dim: int
        specifies the dimensionality of the vector after compression by PQ
    vq_n_centers: int
        specifies the number of centers for the vector quantizer.
        When zero, an optimal value is selected using a heuristic.
        When one, only product quantization is used.
    kmeans_n_iters: int
        specifies the number of iterations searching for kmeans centers
    vq_kmeans_trainset_fraction: float
        specifies the fraction of data to use during iterative kmeans building
        for the vector quantizer. When zero, an optimal value is selected
        using a heuristic.
    pq_kmeans_trainset_fraction: float
        specifies the fraction of data to use during iterative kmeans building
        for the product quantizer. When zero, an optimal value is selected
        using a heuristic.
    pq_kmeans_type: str
        specifies the type of kmeans algorithm to use for PQ training
        possible values: "kmeans", "kmeans_balanced"
    """

    cdef cuvsProductQuantizerParams * params

    def __cinit__(self):
        check_cuvs(cuvsProductQuantizerParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsProductQuantizerParamsDestroy(self.params))

    def __init__(self, *, pq_bits=8, pq_dim=0, vq_n_centers=0,
                 kmeans_n_iters=25, vq_kmeans_trainset_fraction=0.0,
                 pq_kmeans_trainset_fraction=0.0,
                 pq_kmeans_type="kmeans_balanced"):
        if pq_bits not in [4, 5, 6, 7, 8]:
            raise ValueError("pq_bits must be one of [4, 5, 6, 7, 8]")
        self.params.pq_bits = pq_bits
        self.params.pq_dim = pq_dim
        self.params.vq_n_centers = vq_n_centers
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.vq_kmeans_trainset_fraction = vq_kmeans_trainset_fraction
        self.params.pq_kmeans_trainset_fraction = pq_kmeans_trainset_fraction
        pq_kmeans_type_c = PQ_KMEANS_TYPES[pq_kmeans_type]
        self.params.pq_kmeans_type = <cuvsKMeansType>pq_kmeans_type_c

    @property
    def pq_bits(self):
        return self.params.pq_bits

    @property
    def pq_dim(self):
        return self.params.pq_dim

    @property
    def vq_n_centers(self):
        return self.params.vq_n_centers

    @property
    def kmeans_n_iters(self):
        return self.params.kmeans_n_iters

    @property
    def vq_kmeans_trainset_fraction(self):
        return self.params.vq_kmeans_trainset_fraction

    @property
    def pq_kmeans_trainset_fraction(self):
        return self.params.pq_kmeans_trainset_fraction

    @property
    def pq_kmeans_type(self):
        return self.params.pq_kmeans_type


cdef class Quantizer:
    """
    Defines and stores product quantizer upon training

    The quantization is performed by a linear mapping of an interval in the
    float data type to the full range of the quantized int type.
    """
    cdef cuvsProductQuantizer * quantizer

    def __cinit__(self):
        check_cuvs(cuvsProductQuantizerCreate(&self.quantizer))

    def __dealloc__(self):
        check_cuvs(cuvsProductQuantizerDestroy(self.quantizer))

    @property
    def pq_bits(self):
        cdef uint32_t pq_bits
        check_cuvs(cuvsProductQuantizerGetPqBits(self.quantizer, &pq_bits))
        return pq_bits

    @property
    def pq_dim(self):
        cdef uint32_t pq_dim
        check_cuvs(cuvsProductQuantizerGetPqDim(self.quantizer, &pq_dim))
        return pq_dim

    @property
    def pq_codebook(self):
        """
        Returns the PQ codebook
        """
        output = DeviceTensorView()
        cdef cydlpack.DLManagedTensor* pq_codebook_dlpack = \
            <cydlpack.DLManagedTensor*><size_t>output.get_handle()
        check_cuvs(cuvsProductQuantizerGetPqCodebook(self.quantizer,
                                                     pq_codebook_dlpack))
        output.parent = self
        return output

    @property
    def vq_codebook(self):
        """
        Returns the VQ codebook
        """
        output = DeviceTensorView()
        cdef cydlpack.DLManagedTensor* vq_codebook_dlpack = \
            <cydlpack.DLManagedTensor*><size_t>output.get_handle()
        check_cuvs(cuvsProductQuantizerGetVqCodebook(self.quantizer,
                                                     vq_codebook_dlpack))
        output.parent = self
        return output

    @property
    def encoded_dim(self):
        """
        Returns the encoded dimension of the quantized dataset
        """
        cdef uint32_t encoded_dim
        check_cuvs(cuvsProductQuantizerGetEncodedDim(self.quantizer,
                                                     &encoded_dim))
        return encoded_dim

    def __repr__(self):
        return f"product.Quantizer(pq_bits={self.pq_bits}, " \
               f"pq_dim={self.pq_dim})"


@auto_sync_resources
def train(QuantizerParams params, dataset, resources=None):
    """
    Initializes a product quantizer to be used later for quantizing
    the dataset.

    Parameters
    ----------
    params : QuantizerParams object
    dataset : row major device dataset. FP32 or FP64
    {resources_docstring}

    Returns
    -------
    quantizer: cuvs.preprocessing.quantize.product.Quantizer

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import product
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = product.QuantizerParams(pq_bits=8, pq_dim=16)
    >>> quantizer = product.train(params, dataset)
    >>> transformed = product.transform(quantizer, dataset)
    """
    dataset_ai = wrap_array(dataset)

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)

    _check_input_array(dataset_ai,
                       [np.dtype("float32"),
                        np.dtype("float64")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef Quantizer ret = Quantizer()

    check_cuvs(cuvsProductQuantizerTrain(res,
                                         params.params,
                                         dataset_dlpack,
                                         ret.quantizer))

    return ret


@auto_sync_resources
@auto_convert_output
def transform(Quantizer quantizer, dataset, output=None, resources=None):
    """
    Applies product quantization transform to given dataset

    Parameters
    ----------
    quantizer : trained Quantizer object
    dataset : row major device dataset to transform. FP32 or FP64
    output : optional preallocated output memory, on device memory
    {resources_docstring}

    Returns
    -------
    output : transformed dataset quantized into a uint8

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import product
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = product.QuantizerParams(pq_bits=8, pq_dim=16)
    >>> quantizer = product.train(params, dataset)
    >>> transformed = product.transform(quantizer, dataset)
    """

    dataset_ai = wrap_array(dataset)

    _check_input_array(dataset_ai,
                       [np.dtype("float32"),
                        np.dtype("float64")])

    if output is None:
        on_device = hasattr(dataset, "__cuda_array_interface__")
        ndarray = device_ndarray if on_device else np
        encoded_cols = quantizer.encoded_dim
        output = ndarray.empty((dataset_ai.shape[0],
                                encoded_cols), dtype="uint8")

    output_ai = wrap_array(output)
    _check_input_array(output_ai, [np.dtype("uint8")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cydlpack.DLManagedTensor* output_dlpack = \
        cydlpack.dlpack_c(output_ai)

    check_cuvs(cuvsProductQuantizerTransform(res,
                                             quantizer.quantizer,
                                             dataset_dlpack,
                                             output_dlpack))

    return output


@auto_sync_resources
@auto_convert_output
def inverse_transform(Quantizer quantizer, codes, output=None, resources=None):
    """
    Applies product quantization inverse transform to given codes

    Parameters
    ----------
    quantizer : trained Quantizer object
    codes : row major device codes to inverse transform. uint8
    output : optional preallocated output memory, on device memory
    {resources_docstring}

    Returns
    -------
    output : Original dataset reconstructed from quantized codes

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import product
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = product.QuantizerParams(pq_bits=8, pq_dim=16)
    >>> quantizer = product.train(params, dataset)
    >>> transformed = product.transform(quantizer, dataset)
    >>> reconstructed = product.inverse_transform(quantizer, transformed)
    """

    codes_ai = wrap_array(codes)

    _check_input_array(codes_ai, [np.dtype("uint8")])
    pq_cdbk = quantizer.pq_codebook

    if output is None:
        on_device = hasattr(codes, "__cuda_array_interface__")
        ndarray = device_ndarray if on_device else np
        original_cols = quantizer.pq_dim * pq_cdbk.shape[1]
        output = ndarray.empty((codes_ai.shape[0],
                                original_cols), dtype=pq_cdbk.dtype)

    output_ai = wrap_array(output)
    _check_input_array(output_ai, [pq_cdbk.dtype])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* codes_dlpack = \
        cydlpack.dlpack_c(codes_ai)
    cdef cydlpack.DLManagedTensor* output_dlpack = \
        cydlpack.dlpack_c(output_ai)

    check_cuvs(cuvsProductQuantizerInverseTransform(res,
                                                    quantizer.quantizer,
                                                    codes_dlpack,
                                                    output_dlpack))

    return output
