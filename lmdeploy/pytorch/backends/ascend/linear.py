# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from ..linear import LinearImpl, LinearBuilder

class AscendLinearImpl(LinearImpl):
    """ascend implementation api."""
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        from lmdeploy.pytorch.kernels.ascend import linear
        return linear(x, weight, bias)

class AscendLinearBuilder(LinearBuilder):
    """ascend linear implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None)-> AscendLinearImpl:
        """build."""
        return AscendLinearImpl()
