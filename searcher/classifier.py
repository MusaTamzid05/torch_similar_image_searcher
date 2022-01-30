from searcher.data_handler import CustomImageDataset
from torch.utils.data import DataLoader
from searcher.models import Net

import torch
import torch.optim as optim
from torch import nn
import numpy as np

import os
import pickle


class Classifier:
    def __init__(self, data_dir_path):

        self.device = "cuda" if torch.cuda.is_available() else  "cpu"
        self.model_file_name = "model"


        self.train_dataset = CustomImageDataset(dir_path = data_dir_path)
        self.validation_dataset = CustomImageDataset(dir_path = data_dir_path, validation_dataset = True)

        self.model = None
        self.optimizer = None



    def load(self, model_dir):
        print(f"Loading model from {model_dir}")
        pickle_path = os.path.join(model_dir, "label.pickle")
        label_data = None

        with open(pickle_path, "rb") as f:
            label_data = pickle.load(f)

        model_path = os.path.join(model_dir, self.model_file_name)
        num_classes = len(label_data)

        self.model = Net(num_classes = num_classes)

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self._init_optimizer()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("model loaded")


    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.01)

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



    def fit(self, epochs = 100,  batch_size = 16, save_dir_path = "./train_model"):

        self.train_data_loader = DataLoader(self.train_dataset, batch_size = batch_size)
        self.validation_data_loader = DataLoader(self.validation_dataset)

        if self.model is None:
            num_classes = len(self.train_dataset[0]["label"])
            self.model = Net(num_classes = num_classes)
            self.model.to(self.device)
            print("Starting model from scratch")
        else:
            print("using old model")



        if self.optimizer is None:
            self._init_optimizer()

        self.loss_fn = nn.MSELoss(reduction = "mean")

        print("Training started")


        for epoch in range(epochs):
            training_loss  = self._train()
            validation_loss = self._validate()
            print(f"Epoch {epoch} Training loss : {training_loss}  Validation loss : {validation_loss}")


        self._save(save_dir_path = save_dir_path)

    def _save(self, save_dir_path):
        print("Saving")
        if os.path.isdir(save_dir_path) == False:
            os.mkdir(save_dir_path)

        model_save_path = os.path.join(save_dir_path, self.model_file_name)
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            },
             model_save_path)

        label_path = os.path.join(save_dir_path, "label.pickle")

        with open(label_path, "wb") as f:
            pickle.dump(self.train_dataset.label_to_encoder, f)



