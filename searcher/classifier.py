from searcher.data_handler import CustomImageDataset
from torch.utils.data import DataLoader
from searcher.models import Net

import torch
import torch.optim as optim
from torch import nn
import numpy as np


class Classifier:
    def __init__(self, data_dir_path):
        self.train_dataset = CustomImageDataset(dir_path = data_dir_path)
        self.validation_dataset = CustomImageDataset(dir_path = data_dir_path, validation_dataset = True)

        self.device = "cuda" if torch.cuda.is_available() else  "cpu"


    def _train(self):
        training_losses = []

        for train_data in self.train_data_loader:
            x_batch = train_data["src"].to(self.device)
            y_batch = train_data["label"].to(self.device)

            self.model.train()
            yhat = self.model(x_batch)

            loss = self.loss_fn(y_batch, yhat)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            training_losses.append(loss.item())

        return np.mean(training_losses)


    def _validate(self):
        validation_losses = []

        for valid_data in self.validation_data_loader:
            x_batch = valid_data["src"].to(self.device)
            y_batch = valid_data["label"].to(self.device)

            self.model.eval()
            yhat = self.model(x_batch)

            loss = self.loss_fn(y_batch, yhat)
            self.optimizer.zero_grad()

            validation_losses.append(loss.item())

        return np.mean(validation_losses)



    def fit(self, epochs = 100,  batch_size = 16):
        num_classes = len(self.train_dataset[0]["label"])

        self.train_data_loader = DataLoader(self.train_dataset, batch_size = batch_size)
        self.validation_data_loader = DataLoader(self.validation_dataset)

        self.model = Net(num_classes = num_classes)
        self.model.to(self.device)



        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.01)
        self.loss_fn = nn.MSELoss(reduction = "mean")

        print("Training started")


        for epoch in range(epochs):
            training_loss  = self._train()
            validation_loss = self._validate()
            print(f"Epoch {epoch} Training loss : {training_loss}  Validation loss : {validation_loss}")


