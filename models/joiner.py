import torch
import torchvision
import torch.nn.functional as F

from typing import List

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from pos_encoding import PositionalEncodingSine
from utils.misc import NestedTensor


# This one is created when building ResNet (when building each BN layer)
# After the model is built, for each layer _load_from_state_dict is called
class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        # register_buffer -> save to state_dict but don't train

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # TODO: understand it better
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale

        return x * scale + bias


class BackboneBase(nn.Module):
    """
    TODO
    """

    def __init__(self, backbone: nn.Module, train_backbone: bool, 
            num_channels: int):
        """
        TODO
        """

        # super().__init__()
        for name, parameter in backbone.named_parameters():
            # Freeze layers if train_backbone = False,
            # otherwise train only layer 2, 3, and 4
            # TODO: why? which ones are these?
            if not train_backbone or (
                    'layer2' not in name and 
                    'layer3' not in name and 
                    'layer4' not in name):
                parameter.requires_grad_(False)

        return_layers = {'layer4': '0'}

        # IntermediateLayerGetter creates a model constisting of layers
        # until the specified one (or models if there are multiple keys).
        # The result of a call is also a dictionary, where out tensors
        # are values under a specified key (e.g. layer4 output for '0').
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, nested_tensor: NestedTensor):
        xs = self.body(nested_tensor.tensor)
        m = nested_tensor.mask
        assert m is not None

        out = {}
        for name, x in xs.items():
            # TODO: m[None]? [0]?
            new_m = F.interpolate(m[None].float(), 
                        size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, new_m)

        return out


class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm layers.
    """

    def __init__(self, name: str, train_backbone: bool):
        """
        TODO
        """

        # Be careful about dilation, should be False due to None
        # Can use getattr as there's torchvision.models.resnet50
        backbone = getattr(torchvision.models, name)(   
                    pretrained=True, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        super().__init__(backbone, train_backbone, num_channels)

    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []

        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensor.dtype))

        return out, pos  # Returning features + positional encodings


def build_joiner(args):  # TODO: rename to build_joiner
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone)  # No dilation and masks atm
    pos_encoding = PositionalEncodingSine(args.hidden_dim//2, normalize=True)
    model = Joiner(backbone, pos_encoding)
    model.num_channels = backbone.num_channels

    return model