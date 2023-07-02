from torch import nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
  def __init__(self, channels, n_classes):
    super(LeNet_5, self).__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(in_channels = channels, out_channels = 6, kernel_size = 5, stride = 1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size = 2),
        nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size = 2),
        nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1),
        nn.Tanh(),
    )
    
    self.classifier = nn.Sequential(
        nn.Linear(in_features = 120, out_features = 84),
        nn.Tanh(),
        nn.Linear(in_features = 84, out_features = n_classes),
    )

  def forward(self, x):
    x = self.feature_extractor(x) # input = (B, 1, 32, 32) / output = (B, 120, 1, 1)
    x = x.view(-1, 120) # x.shape = (B, 120)
    x = self.classifier(x) # x.shape = (B, n_classes)
    return x

model = LeNet_5(channels = 1, n_classes = 10) # MNIST dataset
