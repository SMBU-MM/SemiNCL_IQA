import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, config=None):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        outdim = 1
        self.config = config
        self.backbone = self._creat_share_backbone(self.config)
        self.backbone1 = self._create_head(self.config)
        self.backbone2 = self._create_head(self.config)
        self.backbone3 = self._create_head(self.config)
        self.backbone4 = self._create_head(self.config)
        self.backbone5 = self._create_head(self.config)
        self.backbone6 = self._create_head(self.config)
        self.backbone7 = self._create_head(self.config)
        self.backbone8 = self._create_head(self.config)
        
        self.backbone_list = [self.backbone1, self.backbone2, self.backbone3, self.backbone4, \
                              self.backbone5, self.backbone6, self.backbone7, self.backbone8]
        
        # euqal to share gamma
        self.share_fc = nn.Linear(outdim, outdim, bias=False)
        nn.init.kaiming_normal_(self.share_fc.weight.data)
        if self.share_fc.bias is not None:
            nn.init.constant_(self.share_fc.bias.data, val=0)
   
    def _creat_share_backbone(self, pretrained = True):
        backbone = nn.Sequential()
        resnet = models.resnet18(pretrained = pretrained)
        # frozen parameter if fz is True
        if self.config.fz:
            for key, param in resnet.named_parameters():
                param.requires_grad = False
                    
        backbone.conv1 = resnet.conv1
        backbone.bn1 = resnet.bn1
        backbone.relu = resnet.relu
        backbone.maxpool = resnet.maxpool
        backbone.layer1 = resnet.layer1
        backbone.layer2 = resnet.layer2
        backbone.layer3 = resnet.layer3
        return backbone
        
    def _create_head(self, config, indim=512, outdim=1, pretrained = True):
        backbone = nn.Sequential()
        resnet = models.resnet18(pretrained=pretrained)
        # frozen parameter if fz is True
        if self.config.fz:
            for key, param in resnet.named_parameters():
                param.requires_grad = False
                    
        backbone.layer4 = resnet.layer4
        backbone.avgpool = resnet.avgpool
        backbone.fc = nn.Linear(indim, outdim, bias=False)
        # initialize fc
        nn.init.kaiming_normal_(backbone.fc.weight.data)
        if backbone.fc.bias is not None:
            nn.init.constant_(backbone.fc.bias.data, val=0)
            
        backbone.fcbn = nn.BatchNorm1d(outdim, affine=False)
        return backbone
    
    def _forward_once(self, backbone, share_x, share_fc):
        # model1
        x = backbone.layer4(share_x)
        x = backbone.avgpool(x).view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        x = backbone.fc(x)
        x = backbone.fcbn(x)
        x = share_fc(x)
        mean = x[:, 0]
        #t = x[:, 1]
        #var = nn.functional.softplus(t)**2
        var = torch.ones_like(mean)**2
        return mean, var
        
    def forward(self, x):
        """
        Forward pass of the network.
        """
        # Forward pass of the shared network.
        share_x = self.backbone(x)
        
        
        # forward pass of each head
        mean_list, var_list = [], []
        for backbone in self.backbone_list:
            mean, var = self._forward_once(backbone, share_x, self.share_fc)
            mean_list.append(mean)
            var_list.append(var)
            
        #calculate ensemble mean and var
        mean_sum = mean_list[0].clone()
        var_sum = var_list[0].clone()
        for i in range(len(mean_list)-1):
            mean_sum +=  mean_list[i+1].clone()
            var_sum +=  var_list[i+1].clone()
        mean_ens = mean_sum/len(mean_list)
        var_ens = var_sum/(len(var_list)*len(var_list))
            
        return mean_list, var_list, mean_ens, var_ens
