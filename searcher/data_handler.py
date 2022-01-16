from torch.utils.data import Dataset
import os
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, dir_path):
        self._load(dir_path = dir_path)

        for path in self.image_paths:
            print(path)




    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

        item = {"src" : image, "dst" : image}

        return item




    def _load(self, dir_path):
        label_names = sorted(os.listdir(dir_path))
        self.image_paths  = []

        for label_name in label_names:
            label_dir_path = os.path.join(dir_path, label_name)
            current_paths = self._load_image_paths(image_dir_path = label_dir_path)

            self.image_paths += current_paths

    def _load_image_paths(self, image_dir_path):
        image_names = os.listdir(image_dir_path)
        image_paths = []

        for image_name in image_names:
            image_path = os.path.join(image_dir_path, image_name)
            image_paths.append(image_path)


        return image_paths










def main():
    dataset = CustomImageDataset(dir_path = "/home/musa/data/images/natural_images/data/natural_images")

    for item in dataset:
        print(item.keys())

if __name__ == "__main__":
    main()

