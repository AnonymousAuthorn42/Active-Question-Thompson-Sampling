import torch
from torch import Tensor
from typing import Any, List, Optional, Type, Union
import torchvision.models as tm
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models._api import  WeightsEnum
from torchvision.models._utils import _ovewrite_named_param


class resnet(tm.ResNet) :
    def set_embSize(self) :
        self.embSize = self.fc.in_features
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        emb = torch.flatten(x, 1)
        x = self.fc(emb)

        return x, emb

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> resnet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = resnet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
    

def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet18_Weights.verify(weights)
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)



def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet34_Weights.verify(weights)
    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)



def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)



def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet101_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)



def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet152_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)

def def_resnet(resnet_type, num_classes):
    if resnet_type == 18:
        model = resnet18(num_classes = num_classes)
    elif resnet_type == 34 :
        model = resnet34(num_classes = num_classes)
    elif resnet_type == 50 :
        model = resnet50(num_classes = num_classes)
    return model
