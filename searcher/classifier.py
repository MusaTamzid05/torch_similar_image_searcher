from searcher.data_handler import CustomImageDataset
from torch.utils.data import DataLoader

class Classifier:
    def __init__(self, data_dir_path):
        self.train_dataset = CustomImageDataset(dir_path = data_dir_path)
        self.validation_dataset = CustomImageDataset(dir_path = data_dir_path, validation_dataset = False)

    def fit(self, epochs = 100,  batch_size = 16):
        self.train_data_loader = DataLoader(self.train_dataset, batch_size = batch_size)
        self.validation_data_loader = DataLoader(self.validation_dataset)

        for epoch  in range(epochs):
            for train_data in self.train_data_loader:
                print(train_data["src"].shape)

