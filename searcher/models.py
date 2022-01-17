import torch.nn as nn
import torch

import torch.nn.functional as F

try:
    from searcher.data_handler import CustomImageDataset
except ModuleNotFoundError:
    from data_handler import CustomImageDataset


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, stride = 2)
        self.conv2 = nn.Conv2d(10, 20, 3, stride = 2)
        self.conv3 = nn.Conv2d(20, 30, 3, stride = 2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(270, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.last_activation = nn.Softmax(dim = 1)





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
        x = self.fc2(x)

        x = self.last_activation(x)
        return x



if __name__ == "__main__":
    dataset = CustomImageDataset(dir_path = "/home/musa/data/images/natural_images/data/natural_images", validation_dataset = True)
    item = dataset[0]["src"]
    #print(item.shape)

    item = item.expand(torch.Size([1, 1, 256, 256]))
    net = Net(num_classes = 5)
    result = net.forward(item)

    print(result.shape)
    print(result)

