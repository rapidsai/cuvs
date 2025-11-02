#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


from __future__ import absolute_import, division, print_function

import sys

import numpy as np


def read_fbin(fname):
    shape = np.fromfile(fname, dtype=np.uint32, count=2)
    if float(shape[0]) * shape[1] * 4 > 2_000_000_000:
        data = np.memmap(fname, dtype=np.float32, offset=8, mode="r").reshape(
            shape
        )
    else:
        data = np.fromfile(fname, dtype=np.float32, offset=8).reshape(shape)
    return data


def write_bin(fname, data):
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)


if len(sys.argv) != 3:
    print(
        "usage: %s input.fbin output.f16bin" % (sys.argv[0]),
        file=sys.stderr,
    )
    sys.exit(-1)

data = read_fbin(sys.argv[1]).astype(np.float16)
write_bin(sys.argv[2], data)
