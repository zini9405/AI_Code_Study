# 코드 참고: https://www.youtube.com/watch?v=ACmuBbuXn20&t=4s

from torch import nn
import torch.nn.functional as F

VGG_types = {'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

class VGGNet(nn.Module):
    def __init__(self, model, in_channels = 3, num_classes = 1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
    
        self.feature_extractor = self.vgg_architecture(VGG_types[model])
        
        self.classifier = nn.Sequential(nn.Linear(in_features = 512 * 7 * 7, out_features = 4096),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(in_features = 4096, out_features = 4096),
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(in_features = 4096, out_features = num_classes)
                                       )
        
    def forward(self, x):
        x = self.feature_extractor(x) # input = (B, 3, 224, 224) / output = (B, 512, 7, 7)
        x = x.view(-1, 512 * 7 * 7) # x.shape = (B, 25088)
        x = self.classifier(x) # x.shape = (B, n_classes)
        x = F.softmax(x, dim = 1)
        return x
    
    def vgg_architecture(self, architecture):
        layers = []
        in_channels = self.in_channels # 초기값
        
        for i in architecture:
            if type(i) == int:
                out_channels = i
                
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
                           nn.BatchNorm2d(i),
                           nn.ReLU(),]
                          
                in_channels = i

            else:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
                
        return nn.Sequential(*layers)

model = VGGNet('VGG13', in_channels=3, num_classes = 1000) # ImageNet dataset
