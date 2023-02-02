import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from torchvision.models import resnet
import torch
from torchvision import models
from torch import nn

from functools import partial


class MANet_encoder(nn.Module):
    def __init__(self, sensor):
        super(MANet_encoder, self).__init__()
        self.name = 'MANet'
        
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        
        if sensor == 'rgb':
            self.firstconv = resnet.conv1
        elif sensor =='thermal':
            self.firstconv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.firstconv.weight.data = torch.unsqueeze(torch.mean(resnet.conv1.weight.data, dim=1), dim=1)
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        # Encoder
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)


        return [e1,e2,e3,e4]