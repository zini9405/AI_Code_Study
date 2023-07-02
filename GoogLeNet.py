# 코드 참고: https://www.youtube.com/watch?v=uQc4Fs7yx5I&t=39s

from torch import nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self, in_channels, aux_classifier = True, num_classes = 1000):
        super(GoogLeNet, self).__init__()
        assert aux_classifier == True or aux_classifier == False
        self.aux_classifier = aux_classifier
        
        self.conv1 = conv_block(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = conv_block(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)

        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)

        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.dropout = nn.Dropout(p = 0.4)
        self.fc1 = nn.Linear(in_features = 1024, out_features = num_classes)

        if self.aux_classifier:
            self.aux1 = InceptionAux(in_channels = 512, num_classes = num_classes)
            self.aux2 = InceptionAux(in_channels = 528, num_classes = num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        if self.aux_classifier and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_classifier and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x) # x.shape = (B, 1024, 7, 7)
        x = self.avgpool(x) # x.shape = (B, 1024, 1, 1)

        x = x.view(x.shape[0], -1) # x.shape = (B, 1024)

        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_classifier and self.training:
            return x, aux1, aux2 # x.shape = aux1 = aux2 = (B, n_classes)
        else:
            return x # x.shape = (B, n_classes)
        
        
        
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(conv_block, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                                       nn.BatchNorm2d(num_features = out_channels),
                                       nn.ReLU(),
                                       )
    def forward(self, x):
        x = self.conv_layer(x)
        return x
    
class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channels = in_channels, out_channels = out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels = in_channels, out_channels = red_3x3, kernel_size=1),
            conv_block(in_channels = red_3x3, out_channels = out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels = in_channels, out_channels = red_5x5, kernel_size=1),
            conv_block(in_channels = red_5x5, out_channels = out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            conv_block(in_channels = in_channels, out_channels = out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # 0_dimension은 batch이므로, 1_dimension인 num_filter를 기준으로 각 branch의 output을 concatenation. 
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            conv_block(in_channels = in_channels, out_channels = 128, kernel_size = 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features = 2048, out_features = 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features = 1024, out_features = num_classes),
        )

    def forward(self, x):
        x = self.conv(x) # x.shape = (B, 128, 4, 4)
        x = x.view(x.shape[0], -1) # x.shape = (B, 2048)
        x = self.fc(x)
        return x
    
model = GoogLeNet(in_channels = 3, aux_classifier = True, num_classes = 1000) # ImageNet dataset

# train 경우 loss, aux_classifier == True
'''
loss_func = nn.CrossEntropyLoss(reduction='sum')        
output_loss = loss_func(output, target)
aux1_loss = loss_func(aux1, target)
aux2_loss = loss_func(aux2, target)
loss = output_loss + 0.3*(aux1_loss + aux2_loss)
'''
# test 경우 loss, aux_classifier == False
  
