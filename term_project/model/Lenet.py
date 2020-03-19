import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2)
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16 * 5 * 5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x