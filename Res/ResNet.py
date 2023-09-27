import torch
import torch.nn as nn

# ResNet 18, 34
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
    
        # BatchNrom에 Bias가 포함되어 있으므로, Conv2d의 Bias를 False로 설정함.
        self.residual_function = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        # Identity mapping (input == output)
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

    # Projection mapping (1 x 1 Conv)
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

# ResNet 50, 101, 152
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.residual_function = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size = 1, stride = 1, bias = False),
        nn.BatchNorm2d(out_channels * BottleNeck.expansion))

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes = 10):
        super().__init__()
    
        self.out_channels = 64

        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = self.out_channels, kernel_size = 7, stride = 2, padding = 3, bias = False),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.conv2 = self.make_layer(block, 64, num_block[0], 1)
        self.conv3 = self.make_layer(block, 128, num_block[1], 2)
        self.conv4 = self.make_layer(block, 256, num_block[2], 2)
        self.conv5 = self.make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

  
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.out_channels, out_channels, stride))
            self.out_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])
