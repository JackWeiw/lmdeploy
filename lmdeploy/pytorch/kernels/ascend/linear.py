# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import dlinfer.ops as ext_ops
from torch import Tensor


def linear(
    x,
    weight: Tensor,
    bias: Optional[Tensor] = None,
):
    """linear operation."""
    return ext_ops.linear(x, weight, bias)
