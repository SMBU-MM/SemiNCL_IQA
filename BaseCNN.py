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
        self.backbone5 = models.resnet18(pretrained=True)
        self.backbone6 = models.resnet18(pretrained=True)
        self.backbone7 = models.resnet18(pretrained=True)
        self.backbone8 = models.resnet18(pretrained=True)

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
            # Freeze all previous layers.
            for key, param in self.backbone5.named_parameters():
                param.requires_grad = False
            # Freeze all previous layers.
            for key, param in self.backbone6.named_parameters():
                param.requires_grad = False
            # Freeze all previous layers.
            for key, param in self.backbone7.named_parameters():
                param.requires_grad = False
            # Freeze all previous layers.
            for key, param in self.backbone8.named_parameters():
                param.requires_grad = False
                
        outdim = 1
        self.backbone1.fc = nn.Linear(512, outdim, bias=False)
        self.backbone2.fc = nn.Linear(512, outdim, bias=False)
        self.backbone3.fc = nn.Linear(512, outdim, bias=False)
        self.backbone4.fc = nn.Linear(512, outdim, bias=False)
        self.backbone5.fc = nn.Linear(512, outdim, bias=False)
        self.backbone6.fc = nn.Linear(512, outdim, bias=False)
        self.backbone7.fc = nn.Linear(512, outdim, bias=False)
        self.backbone8.fc = nn.Linear(512, outdim, bias=False)

        # Initialize the fc layers.
        nn.init.kaiming_normal_(self.backbone1.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone2.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone3.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone4.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone5.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone6.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone7.fc.weight.data)
        nn.init.kaiming_normal_(self.backbone8.fc.weight.data)

        if self.backbone1.fc.bias is not None:
            nn.init.constant_(self.backbone1.fc.bias.data, val=0)
            nn.init.constant_(self.backbone2.fc.bias.data, val=0)
            nn.init.constant_(self.backbone3.fc.bias.data, val=0)
            nn.init.constant_(self.backbone4.fc.bias.data, val=0)
            nn.init.constant_(self.backbone5.fc.bias.data, val=0)
            nn.init.constant_(self.backbone6.fc.bias.data, val=0)
            nn.init.constant_(self.backbone7.fc.bias.data, val=0)
            nn.init.constant_(self.backbone8.fc.bias.data, val=0)
        
        # euqal to share gamma
        self.share_fc = nn.Linear(outdim, outdim, bias=False)
        nn.init.kaiming_normal_(self.share_fc.weight.data)
        if self.share_fc.bias is not None:
            nn.init.constant_(self.share_fc.bias.data, val=0)

        self.backbone1.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone2.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone3.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone4.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone5.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone6.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone7.fcbn = nn.BatchNorm1d(outdim, affine=False)
        self.backbone8.fcbn = nn.BatchNorm1d(outdim, affine=False)

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
        x1 = F.normalize(x1, p=2, dim=1)
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
        x2 = F.normalize(x2, p=2, dim=1)
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
        x3 = F.normalize(x3, p=2, dim=1)
        x3 = self.backbone3.fc(x3)
        x3 = self.backbone3.fcbn(x3)
        x3 = self.share_fc(x3)

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

        # model 5
        x5 = self.backbone5.layer3(x_share)
        x5 = self.backbone5.layer4(x5)
        x5 = self.backbone5.avgpool(x5).view(x.size(0), -1)
        x5 = F.normalize(x5, p=2, dim=1)
        x5 = self.backbone5.fc(x5)
        x5 = self.backbone5.fcbn(x5)
        x5 = self.share_fc(x5)
        mean5 = x5[:, 0]
        #t = x4[:, 1]
        #var4 = nn.functional.softplus(t)**2
        var5 = torch.ones_like(mean5)**2

        # model 6
        x6 = self.backbone6.layer3(x_share)
        x6 = self.backbone6.layer4(x6)
        x6 = self.backbone6.avgpool(x6).view(x.size(0), -1)
        x6 = F.normalize(x6, p=2, dim=1)
        x6 = self.backbone6.fc(x6)
        x6 = self.backbone6.fcbn(x6)
        x6 = self.share_fc(x6)
        mean6 = x6[:, 0]
        #t = x4[:, 1]
        #var4 = nn.functional.softplus(t)**2
        var6 = torch.ones_like(mean6)**2
        
        # model 7
        x7 = self.backbone7.layer3(x_share)
        x7 = self.backbone7.layer4(x7)
        x7 = self.backbone7.avgpool(x7).view(x.size(0), -1)
        x7 = F.normalize(x7, p=2, dim=1)
        x7 = self.backbone7.fc(x7)
        x7 = self.backbone7.fcbn(x7)
        x7 = self.share_fc(x7)
        mean7 = x7[:, 0]
        #t = x4[:, 1]
        #var4 = nn.functional.softplus(t)**2
        var7 = torch.ones_like(mean7)**2

        # model 8
        x8 = self.backbone8.layer3(x_share)
        x8 = self.backbone8.layer4(x8)
        x8 = self.backbone8.avgpool(x8).view(x.size(0), -1)
        x8 = F.normalize(x8, p=2, dim=1)
        x8 = self.backbone8.fc(x8)
        x8 = self.backbone8.fcbn(x8)
        x8 = self.share_fc(x8)
        mean8 = x8[:, 0]
        #t = x4[:, 1]
        #var4 = nn.functional.softplus(t)**2
        var8 = torch.ones_like(mean8)**2

        return [mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8], [var1, var2, var3, var4, var5, var6, var7, var8], \
               (mean1+mean2+mean3+mean4+mean5+mean6+mean7+mean8)/8, (var1+var2+var3+var4+var5+var6+var7+var8)/(8*8)
   
