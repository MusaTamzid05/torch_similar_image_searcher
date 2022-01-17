from searcher.data_handler import CustomImageDataset
from torch.utils.data import DataLoader
from searcher.models import Net

import torch
import torch.optim as optim
from torch import nn


class Classifier:
    def __init__(self, data_dir_path):
        self.train_dataset = CustomImageDataset(dir_path = data_dir_path)
        self.validation_dataset = CustomImageDataset(dir_path = data_dir_path, validation_dataset = False)

        self.device = "cuda" if torch.cuda.is_available() else  "cpu"

    def fit(self, epochs = 100,  batch_size = 16):
        num_classes = len(self.train_dataset[0]["label"])

        self.train_data_loader = DataLoader(self.train_dataset, batch_size = batch_size)
        self.validation_data_loader = DataLoader(self.validation_dataset)

        model = Net(num_classes = num_classes)
        model.to(self.device)



        optimizer = optim.Adam(model.parameters(), lr = 0.01)
        loss_fn = nn.MSELoss(reduction = "mean")

        for epoch in range(epochs):
            for train_data in self.train_data_loader:
                x_batch = train_data["src"].to(self.device)
                y_batch = train_data["label"].to(self.device)

                model.train()
                yhat = model(x_batch)

                print(yhat)



