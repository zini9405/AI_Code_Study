# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/inceptionv4.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic ConV2d

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
      
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, bias = False, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())
    
    def forward(self, x):
        x = self.conv(x)
        return x

# InceptionV4

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Sequential(BasicConv2d(3, 32, 3, stride = 2, padding = 0),
                                   BasicConv2d(32, 32, 3, stride = 1, padding = 0),
                                   BasicConv2d(32, 64, 3, stride = 1, padding = 1))

        self.branch3x3_conv = BasicConv2d(64, 96, 3, stride = 2, padding = 0)
        self.branch3x3_pool = nn.MaxPool2d(4, stride = 2, padding = 1)
        self.branch7x7a = nn.Sequential(BasicConv2d(160, 64, 1, stride = 1, padding = 0),
                                        BasicConv2d(64, 96, 3, stride = 1, padding = 0))

        self.branch7x7b = nn.Sequential(BasicConv2d(160, 64, 1, stride = 1, padding = 0),
                                        BasicConv2d(64, 64, (7,1), stride = 1, padding = (3,0)),
                                        BasicConv2d(64, 64, (1,7), stride = 1, padding = (0,3)),
                                        BasicConv2d(64, 96, 3, stride = 1, padding = 0))

        self.branchpoola = BasicConv2d(192, 192, 3, stride = 2, padding = 0)
        self.branchpoolb = nn.MaxPool2d(4, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat((self.branch3x3_conv(x), self.branch3x3_pool(x)), dim=1)
        x = torch.cat((self.branch7x7a(x), self.branch7x7b(x)), dim=1)
        x = torch.cat((self.branchpoola(x), self.branchpoolb(x)), dim=1)
        return x

# Inception_Resnet

class Inception_Resnet_A(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
      
        self.branch1x1 = BasicConv2d(in_channels, 32, 1, stride = 1, padding = 0)
        self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 32, 1, stride = 1, padding = 0),
                                       BasicConv2d(32, 32, 3, stride = 1, padding = 1))
        self.branch3x3stack = nn.Sequential(BasicConv2d(in_channels, 32, 1, stride = 1, padding = 0),
                                            BasicConv2d(32, 48, 3, stride = 1, padding = 1),
                                            BasicConv2d(48, 64, 3, stride = 1, padding = 1))
        self.reduction1x1 = nn.Conv2d(128, 384, 1, stride = 1, padding = 0)
        self.shortcut = nn.Conv2d(in_channels, 384, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = torch.cat((self.branch1x1(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)
        x = self.reduction1x1(x)
        x = self.bn(x_shortcut + x)
        x = self.relu(x)
        return x

class Inception_Resnet_B(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, 1, stride=1, padding=0)
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, 128, 1, stride=1, padding=0),
            BasicConv2d(128, 160, (1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, (7,1), stride=1, padding=(3,0))
        )

        self.reduction1x1 = nn.Conv2d(384, 1152, 1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(in_channels, 1152, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1152)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = torch.cat((self.branch1x1(x), self.branch7x7(x)), dim=1)
        x = self.reduction1x1(x) * 0.1
        x = self.bn(x + x_shortcut)
        x = self.relu(x)
        return x

class Inception_Resnet_C(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, 1, stride=1, padding=0)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, 1, stride=1, padding=0),
            BasicConv2d(192, 224, (1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, (3,1), stride=1, padding=(1,0))
        )

        self.reduction1x1 = nn.Conv2d(448, 2144, 1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(in_channels, 2144, 1, stride=1, padding=0) # 2144
        self.bn = nn.BatchNorm2d(2144)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x = torch.cat((self.branch1x1(x), self.branch3x3(x)), dim=1)
        x = self.reduction1x1(x) * 0.1
        x = self.bn(x_shortcut + x)
        x = self.relu(x)
        return x
      
#  Inception-Resnet-V2; k=256, l=256, m=384, n=384

class ReductionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super().__init__()

        self.branchpool = nn.MaxPool2d(3, 2)
        self.branch3x3 = BasicConv2d(in_channels, n, 3, stride=2, padding=0)
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, k, 1, stride=1, padding=0),
            BasicConv2d(k, l, 3, stride=1, padding=1),
            BasicConv2d(l, m, 3, stride=2, padding=0)
        )

        self.output_channels = in_channels + n + m

    def forward(self, x):
        x = torch.cat((self.branchpool(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)
        return x

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branchpool = nn.MaxPool2d(3, 2)
        self.branch3x3a = nn.Sequential(
            BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
            BasicConv2d(256, 384, 3, stride=2, padding=0)
        )
        self.branch3x3b = nn.Sequential(
            BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
            BasicConv2d(256, 288, 3, stride=2, padding=0)
        )
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
            BasicConv2d(256, 288, 3, stride=1, padding=1),
            BasicConv2d(288, 320, 3, stride=2, padding=0)
        )

    def forward(self, x):
        x = torch.cat((self.branchpool(x), self.branch3x3a(x), self.branch3x3b(x), self.branch3x3stack(x)), dim=1)
        return x



class InceptionResNetV2(nn.Module):
    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, num_classes=10):
        super().__init__()
      
        blocks = []
      
        blocks.append(Stem())
      
        for i in range(A):
            blocks.append(Inception_Resnet_A(384))
        blocks.append(ReductionA(384, k, l, m, n))
        for i in range(B):
            blocks.append(Inception_Resnet_B(1152))
        blocks.append(ReductionB(1152))
        for i in range(C):
            blocks.append(Inception_Resnet_C(2144))

        self.features = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
      
        # drop out
        self.dropout = nn.Dropout2d(0.2)
        self.linear = nn.Linear(2144, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x



# x = torch.randn((3, 3, 299, 299))
# model = InceptionResNetV2(10, 20, 10)
# output_Stem = model(x)
# print('Input size:', x.size())
# print('Stem output size:', output_Stem.size())























