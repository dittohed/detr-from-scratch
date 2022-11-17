import torch
import torchvision

from torch import Tensor


class NestedTensor():
    """
    Wraps up the tensor and it's corresponding mask
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

    @classmethod
    def from_tensors(cls, tensors: list[Tensor]):
        """
        Converts a list of tensors (containing images of different sizes)
        to NestedTensor.
        """

        assert tensors[0].ndim == 3, 'Only 3 channel imgs are supported'

        max_size = max_by_axis([list(img.shape) for img in tensors])
        batch_shape = [len(tensors)] + max_size
        b, c, h, w = batch_shape

        dtype = tensors[0].dtype
        device = tensors[0].device

        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        # Paste img starting with upper left corner
        # Set False for non-padded pixels
        for img, img_padded, m in zip(tensors, tensor, mask):
            img_padded[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False

        return cls(tensor, mask)        


def collate_fn(batch):
    """
    Converts a sequence of (img, target) pairs to a pair of (imgs, targets)
    where imgs are wrapped up with NestedTensor.
    """

    batch = list(zip(*batch))
    batch[0] = NestedTensor.from_tensors(batch[0])
    return tuple(batch)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def max_by_axis(shapes: list[list[int]]) -> list[int]:
    """
    Given shapes of tensors (images), calculate maxima for each axis. 
    """

    maxes = shapes[0]
    for shape in shapes[1:]:
        for dim, pxs in enumerate(shape):
            maxes[dim] = max(maxes[dim], pxs)

    return maxes