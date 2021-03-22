import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, config=None):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        self.backbone1 = models.resnet18(pretrained=True)
        self.backbone2 = models.resnet18(pretrained=True)
        self.backbone3 = models.resnet18(pretrained=True)
        self.backbone4 = models.resnet18(pretrained=True)
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
        self.backbone1.fc = nn.Linear(512, outdim, bias=False)
        self.backbone2.fc = nn.Linear(512, outdim, bias=False)
        self.backbone3.fc = nn.Linear(512, outdim, bias=False)
        self.backbone4.fc = nn.Linear(512, outdim, bias=False)

        # Initialize the fc layers.
        nn.init.kaiming_normal_(self.backbone1.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone2.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone3.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone4.fc.weight.data)
        if self.backbone1.fc.bias is not None:
            nn.init.constant_(self.backbone1.fc1.bias.data, val=0)
            nn.init.constant_(self.backbone2.fc1.bias.data, val=0)
            nn.init.constant_(self.backbone3.fc1.bias.data, val=0)
            nn.init.constant_(self.backbone4.fc1.bias.data, val=0)
        
        # euqal to share gamma
        self.share_fc = nn.Linear(outdim, outdim, bias=False)
        nn.init.kaiming_normal_(self.share_fc.weight.data)
        if self.share_fc.bias is not None:
            nn.init.constant_(self.share_fc.bias.data, val=0)

        self.backbone1.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone2.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone3.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone4.fcbn = nn.BatchNorm1d(outdim, affine=False)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self.backbone1.conv1(x)
        x = self.backbone1.bn1(x)
        x = self.backbone1.relu(x)
        x = self.backbone1.maxpool(x)
        x = self.backbone1.layer1(x)
        x_share = self.backbone1.layer2(x)
       
        # model1
        x1 = self.backbone1.layer3(x_share)
        x1 = self.backbone1.layer4(x1)
        x1 = self.backbone1.avgpool(x1).view(x.size(0), -1)
        x1 = self.backbone1.fc(x1)
        x1 = self.backbone1.fcbn(x1)
        x1 = self.share_fc(x1)
        mean1 = x1[:, 0]
        #t = x1[:, 1]
        #var1 = nn.functional.softplus(t)**2
        var1 = torch.ones_like(mean1)**2
        
        # model 2
        x2 = self.backbone2.layer3(x_share)
        x2 = self.backbone2.layer4(x2)
        x2 = self.backbone2.avgpool(x2).view(x.size(0), -1)
        #x2 = F.normalize(x2, p=2, dim=1)
        x2 = self.backbone2.fc(x2)
        x2 = self.backbone2.fcbn(x2)
        x2 = self.share_fc(x2)
        mean2 = x2[:, 0]
        #t = x2[:, 1]
        #var2 = nn.functional.softplus(t)**2
        var2 = torch.ones_like(mean2)**2

        # model 3
        x3 = self.backbone3.layer3(x_share)
        x3 = self.backbone3.layer4(x3)
        x3 = self.backbone3.avgpool(x3).view(x.size(0), -1)
        #x3 = F.normalize(x3, p=2, dim=1)
        x3 = self.backbone3.fc(x3)
        x3 = self.backbone3.fcbn(x3)
        mean3 = x3[:, 0]
        #t = x3[:, 1]
        #var3 = nn.functional.softplus(t)**2
        var3 = torch.ones_like(mean3)**2
        
        # model 4
        x4 = self.backbone4.layer3(x_share)
        x4 = self.backbone4.layer4(x4)
        x4 = self.backbone4.avgpool(x4).view(x.size(0), -1)
        x4 = F.normalize(x4, p=2, dim=1)
        x4 = self.backbone4.fc(x4)
        x4 = self.backbone4.fcbn(x4)
        x4 = self.share_fc(x4)
        mean4 = x4[:, 0]
        #t = x4[:, 1]
        #var4 = nn.functional.softplus(t)**2
        var4 = torch.ones_like(mean4)**2
        return [mean1, mean2, mean3, mean4], [var1, var2, var3, var4]
   
