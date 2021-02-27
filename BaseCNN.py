import torch
from torch import Tensor
import torch.nn as nn
from torchvision import models
from torch.nn.utils import weight_norm
from typing import Callable, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        ds: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.ds = ds
        self.conv1 = weight_norm(conv3x3(inplanes, planes, stride))
        self.bn1 = weight_norm(norm_layer(planes))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = weight_norm(conv3x3(planes, planes))
        self.bn2 = weight_norm(norm_layer(planes))
        if ds == True:
            self.downsample = nn.Sequential(weight_norm(conv1x1(inplanes, planes, stride)),
            weight_norm(nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True))
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
    
    
class BaseCNN(nn.Module):
    def __init__(self, config=None):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        self.backbone1 = models.resnet18(pretrained=True)
        layer4_state_dict = self.backbone1.layer4.state_dict()
        self.backbone1.layer4 = self._layer4(layer4_state_dict)
        self.backbone2 = models.resnet18(pretrained=True)
        self.backbone2.layer4 = self._layer4(layer4_state_dict)
        self.backbone3 = models.resnet18(pretrained=True)
        self.backbone3.layer4 = self._layer4(layer4_state_dict)
        self.backbone4 = models.resnet18(pretrained=True)
        self.backbone4.layer4 = self._layer4(layer4_state_dict)
        if self.config.fz:
        #if True:
            # Freeze all previous layers.
            for key, param in self.backbone1.named_parameters():
                param.requires_grad = False
            # Freeze all previous layers.
            for key, param in self.backbone2.named_parameters():
                param.requires_grad = False
            # Freeze all previous layers.
            for key, param in self.backbone3.named_parameters():
                param.requires_grad = False
            # Freeze all previous layers.
            for key, param in self.backbone4.named_parameters():
                param.requires_grad = False
        outdim = 1
        self.backbone1.fc = weight_norm(nn.Linear(512, outdim, bias=False))
        self.backbone2.fc = weight_norm(nn.Linear(512, outdim, bias=False))
        self.backbone3.fc = weight_norm(nn.Linear(512, outdim, bias=False))
        self.backbone4.fc = weight_norm(nn.Linear(512, outdim, bias=False))
        # Initialize the weight_v layers
        nn.init.kaiming_normal_(self.backbone1.fc.weight_v.data)
        nn.init.kaiming_normal_(self.backbone2.fc.weight_v.data)
        nn.init.kaiming_normal_(self.backbone3.fc.weight_v.data)
        nn.init.kaiming_normal_(self.backbone4.fc.weight_v.data)
        # Set weight_g with a constant value 1
        nn.init.constant_(self.backbone1.fc.weight_v.data, val=1)
        nn.init.constant_(self.backbone2.fc.weight_v.data, val=1)
        nn.init.constant_(self.backbone3.fc.weight_v.data, val=1)
        nn.init.constant_(self.backbone4.fc.weight_v.data, val=1)
        
        ##########################################################
        # set all weight_g non-trainable
        ##########################################################
        for key, param in self.backbone1.named_parameters():
            if ".weight_g" in key:
                param.requires_grad = False
        # Freeze all previous layers.
        for key, param in self.backbone2.named_parameters():
            if ".weight_g" in key:
                param.requires_grad = False
        # Freeze all previous layers.
        for key, param in self.backbone3.named_parameters():
            if ".weight_g" in key:
                param.requires_grad = False
        # Freeze all previous layers.
        for key, param in self.backbone4.named_parameters():
            if ".weight_g" in key:
                param.requires_grad = False
            
    def _layer4(self, state_dict):
        layer4 = nn.Sequential(BasicBlock(256, 512, ds = 1, stride=2, dilation=False),
                               BasicBlock(512, 512, ds = 0, stride=1, dilation=False))
        new_state_dict = layer4.state_dict()
        for key in new_state_dict.keys():
            if ".weight_g" in key:
                new_state_dict[key] = torch.ones_like(new_state_dict[key])
            elif ".weight_v" in key:
                new_state_dict[key] = state_dict[key.replace(".weight_v", ".weight")]
            else:
                new_state_dict[key] = state_dict[key]
        layer4.load_state_dict(new_state_dict)
        return layer4
        
    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self.backbone1.conv1(x)
        x = self.backbone1.bn1(x)
        x = self.backbone1.relu(x)
        x = self.backbone1.maxpool(x)
        x = self.backbone1.layer1(x)
        x = self.backbone1.layer2(x)
        x_share = self.backbone1.layer3(x)
        
        # model1
        x1 = self.backbone1.layer4(x_share)
        x1 = self.backbone1.avgpool(x1).view(x.size(0), -1)
        x1 = self.backbone1.fc(x1)
        mean1 = x1[:, 0]
        var1 = torch.ones_like(mean1)**2
        
        # model 2
        x2 = self.backbone2.layer4(x_share)
        x2 = self.backbone2.avgpool(x2).view(x.size(0), -1)
        x2 = self.backbone2.fc(x2)
        mean2 = x2[:, 0]
        var2 = torch.ones_like(mean2)**2

        # model 3
        x3 = self.backbone3.layer4(x_share)
        x3 = self.backbone3.avgpool(x3).view(x.size(0), -1)
        x3 = self.backbone3.fc(x3)
        mean3 = x3[:, 0]
        var3 = torch.ones_like(mean3)**2
        
        # model 4
        x4 = self.backbone4.layer4(x_share)
        x4 = self.backbone4.avgpool(x4).view(x.size(0), -1)
        x4 = self.backbone4.fc(x4)
        mean4 = x4[:, 0]
        var4 = torch.ones_like(mean4)**2
        return mean1, var1, mean2, var2, mean3, var3, mean4, var4

BaseCNN() 