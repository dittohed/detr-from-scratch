from torch import Tensor
from typing import Optional, List

class NestedTensor(object):
    """
    Contains the tensor and it's corresponding mask
    stemming from padding (to ignore some values when calculating
    attention values).
    """

    def __init__(self, tensors, mask: Tensor = None):
        self.tensors = tensors  # TODO: tensor vs. tensors?
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)