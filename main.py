from searcher.models import Net
from searcher.data_handler import CustomImageDataset
import numpy as np
import torch

from searcher.classifier import Classifier


if __name__ == "__main__":
    cls = Classifier(data_dir_path = "/home/musa/data/images/natural_images/data/natural_images")
    cls.fit()






