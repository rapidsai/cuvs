#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsGetLastErrorText


class CuvsException(Exception):
    pass


def get_last_error_text():
    """ returns the last error description from the cuvs c-api """
    cdef const char* c_err = cuvsGetLastErrorText()
    if c_err is NULL:
        return
    cdef bytes err = c_err
    return err.decode("utf8", "ignore")


def check_cuvs(status: cuvsError_t):
    """ Converts a status code into an exception """
    if status == cuvsError_t.CUVS_ERROR:
        raise CuvsException(get_last_error_text())
