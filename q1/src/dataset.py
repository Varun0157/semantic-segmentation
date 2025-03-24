import os
import logging
from enum import Enum
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2


class Mode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RoadDataset(Dataset):
    def __init__(self, mode: Mode, root_dir: str):
        self.mode = mode

        self.images_dir = os.path.join(root_dir, mode.value, "images")
        self.image_files = sorted(os.listdir(self.images_dir))

        self.labels_dir = os.path.join(root_dir, mode.value, "labels")
        self.label_files = sorted(os.listdir(self.labels_dir))

        assert len(self.image_files) == len(
            self.label_files
        ), "number of images does not match the number of labels"
        assert (
            len(self.image_files) > 0
        ), "no image files found in the training directory"

        logging.info(f"loaded {len(self.image_files)} {mode.value} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.images_dir, self.image_files[index])
        label_path = os.path.join(self.labels_dir, self.label_files[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(image)
        assert type(image_tensor) is torch.Tensor

        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        label = label[..., 0]
        label_tensor = torch.from_numpy(label).unsqueeze(-1)

        return image_tensor, label_tensor
