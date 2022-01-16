import torch.nn as nn
import torch

import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, stride = 2)
        self.conv2 = nn.Conv2d(10, 20, 3, stride = 2)
        self.conv3 = nn.Conv2d(20, 30, 3, stride = 2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(270, 100)





    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)


        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x
