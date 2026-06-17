# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""cuTile Python vector-add kernel used by the embedded-cubin example test."""

from __future__ import annotations

import cuda.tile as ct

TILE_SIZE = 256


@ct.kernel
def vector_add(a, b, c, TILE_SIZE: ct.Constant):
    bid = ct.bid(0)
    ta = ct.load(a, bid, TILE_SIZE)
    tb = ct.load(b, bid, TILE_SIZE)
    ct.store(c, bid, ta + tb)
