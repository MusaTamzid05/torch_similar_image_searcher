from torch.utils.data import Dataset
import os
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, dir_path):
        self._load(dir_path = dir_path)

        print(len(self.X), len(self.y))



    def _load(self, dir_path):
        label_names = sorted(os.listdir(dir_path))
        self.X, self.y = [], []

        for label_name in label_names:
            label_dir_path = os.path.join(dir_path, label_name)
            images = self._load_images(image_dir_path = label_dir_path)

            self.X += images
            self.y += images

    def _load_images(self, image_dir_path):
        image_names = os.listdir(image_dir_path)
        images = []

        for image_name in image_names:
            image_path = os.path.join(image_dir_path, image_name)
            image = cv2.imread(image_path)
            images.append(image)


        return images










def main():
    dataset = CustomImageDataset(dir_path = "/home/musa/data/images/natural_images/data/natural_images")

if __name__ == "__main__":
    main()

