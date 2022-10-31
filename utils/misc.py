from torch import Tensor


class NestedTensor(object):
    """
    Contains the tensor and it's corresponding mask
    stemming from padding (to ignore some values when calculating
    attention values).
    """

    def __init__(self, tensor, mask: Tensor = None):
        self.tensor = tensor
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensor.to(device)

        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None

        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensor, self.mask

    def __repr__(self):
        return str(self.tensor)