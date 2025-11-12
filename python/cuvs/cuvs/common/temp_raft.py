#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3


# This file has code that will be upstreamed to RAFT

import functools

from pylibraft.common import DeviceResources

_resources_param_string = """
     handle : Optional RAFT resource handle for reusing CUDA resources.
        If a handle isn't supplied, CUDA resources will be
        allocated inside this function and synchronized before the
        function exits. If a handle is supplied, you will need to
        explicitly synchronize yourself by calling `handle.sync()`
        before accessing the output.
""".strip()


def auto_sync_resources(f):
    """
    This is identical to auto_sync_handle except for the proposed name change.
    """

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_resources = resources is None
        resources = resources if resources is not None else DeviceResources()

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_resources:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        resources_docstring=_resources_param_string
    )
    return wrapper
