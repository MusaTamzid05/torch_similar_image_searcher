from searcher.models import Net
from searcher.data_handler import CustomImageDataset
import numpy as np
import torch


if __name__ == "__main__":
    dataset = CustomImageDataset(dir_path = "/home/musa/data/images/natural_images/data/natural_images", validation_dataset = True)
    item = dataset[0]["src"]

    item = item.expand(torch.Size([1, 1, 256, 256]))
    net = Net()
    result = net.forward(item)

    print(result.shape)






