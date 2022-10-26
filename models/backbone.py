import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import NestedTensor


class FrozenBatchNorm2d(torch.nn.Module):
    pass


class BackboneBase(nn.Module):
    """
    TODO
    """

    def __init__(self, backbone: nn.Module, train_backbone: bool,
            num_channels: int, return_interm_layers: bool):
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

        if return_interm_layers:
            pass  # TODO
        else:
            return_layers = {'layer4': '0'}

        # IntermediateLayerGetter creates a model constisting of layers
        # until the specified one (or models if there are multiple keys).
        # The result of a call is also a dictionary, where out tensors
        # are values under a specified key (e.g. layer4 output for '0').
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, nested_tensor: NestedTensor):
        xs = self.body(nested_tensor.tensors)
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

    def __init__(self, name: str, train_backbone: bool,
            return_interm_layers: bool):
        """
        TODO
        """

        # Be careful about dilation, should be False due to None
        # Can use getattr as there's torchvision.models.resnet50
        backbone = getattr(torchvision.models, name)(   
                    pretrained=True, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        super().__init__(backbone, train_backbone, 
                         num_channels, return_interm_layers)

def build(args):
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone,
                        return_interm_layers)  # No dilation atm