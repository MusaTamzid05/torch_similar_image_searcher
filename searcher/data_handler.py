from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from torchvision import transforms

import torch


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image


class Normalize:
    def __call__(self, image):
        image = image.astype("float32") / 255.0
        return image

class ToTensor:
    def __call__(self, image):
        image = np.expand_dims(image, axis = 0)
        return torch.from_numpy(image)


class CustomImageDataset(Dataset):
    def __init__(self,  dir_path, validation_dataset = False, size = 256):
        self._load(dir_path = dir_path, validation_dataset = validation_dataset)

        self.transforms = transforms.Compose([Resize(size = size), Normalize(), ToTensor()])





    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


        processed_image = self.transforms(image)


        item = {"src" : processed_image , "dst" : processed_image}

        return item




    def _load(self, dir_path, validation_dataset):
        label_names = sorted(os.listdir(dir_path))
        image_paths  = []

        for label_name in label_names:
            label_dir_path = os.path.join(dir_path, label_name)
            current_paths = self._load_image_paths(image_dir_path = label_dir_path)

            image_paths += current_paths

        train_index = int(0.8 * len(image_paths))

        if validation_dataset == False:
            self.image_paths = image_paths[:train_index]
        else:
            self.image_paths = image_paths[train_index:]


    def _load_image_paths(self, image_dir_path):
        image_names = os.listdir(image_dir_path)
        image_paths = []

        for image_name in image_names:
            image_path = os.path.join(image_dir_path, image_name)
            image_paths.append(image_path)

        return image_paths



def main():
    dataset = CustomImageDataset(dir_path = "/home/musa/data/images/natural_images/data/natural_images", validation_dataset = True)

    for item in dataset:
        print(item["src"].shape)

if __name__ == "__main__":
    main()

