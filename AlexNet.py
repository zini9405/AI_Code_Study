from torch import nn
import torch.nn.functional as F

class Alexnet(nn.Module):
  def __init__(self, channels, n_classes):
    super(Alexnet, self).__init__()
    self.feature_extractor = nn.Sequential(
        # layer 1
        nn.Conv2d(in_channels = channels, out_channels = 96, kernel_size = 11, stride = 4),
        nn.ReLU(),
        nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
        nn.MaxPool2d(kernel_size = 3, stride = 2),
        
        # layer 2
        nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2),
        nn.ReLU(),
        nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),
        nn.MaxPool2d(kernel_size = 3, stride = 2)

        # layer 3
        nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),

        # layer 4
        nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),

        # layer 5
        nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2)
    )
    
    self.classifier = nn.Sequential(
        nn.Linear(in_features = 256 * 6 * 6, out_features = 4096),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(in_features = 4096, out_features = 4096),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(in_features = 4096, out_features = n_classes),
    )

  def forward(self, x):
    x = self.feature_extractor(x) # input = (B, 3, 227, 227) / output = (B, 256, 6, 6)
    x = x.view(-1, 256 * 6 * 6) # x.shape = (B, 4096)
    x = self.classifier(x) # x.shape = (B, n_classes)
    x = F.softmax(x, dim = 1)
    return x

model = AlexNet(channels = 3, n_classes = 1000) # ImageNet dataset
