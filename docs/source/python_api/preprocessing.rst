Preprocessing
=============

.. role:: py(code)
   :language: python
   :class: highlight

Binary Quantizer
################

.. autofunction:: cuvs.preprocessing.quantize.binary.transform

Product Quantizer
#################

.. autoclass:: cuvs.preprocessing.quantize.pq.Quantizer
    :members:

.. autoclass:: cuvs.preprocessing.quantize.pq.QuantizerParams
    :members:

.. autofunction:: cuvs.preprocessing.quantize.pq.build

.. autofunction:: cuvs.preprocessing.quantize.pq.transform

.. autofunction:: cuvs.preprocessing.quantize.pq.inverse_transform

Scalar Quantizer
################

.. autoclass:: cuvs.preprocessing.quantize.scalar.Quantizer
    :members:

.. autoclass:: cuvs.preprocessing.quantize.scalar.QuantizerParams
    :members:

.. autofunction:: cuvs.preprocessing.quantize.scalar.train

.. autofunction:: cuvs.preprocessing.quantize.scalar.transform

.. autofunction:: cuvs.preprocessing.quantize.scalar.inverse_transform
