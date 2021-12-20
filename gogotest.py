import torch
from torch import nn


class HisNet(nn.Module):
    def __init__(self):
        super(HisNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 10, 3),
            nn.Flatten(),
            # nn.Linear(128 * 3 * 3, 512),
            nn.Linear(10 * 1 * 1, 5)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


