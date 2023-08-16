from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os
import csv
import yaml
import cv2
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as Ap

from constants import DEFAULT_CONFIG_PATH, Config
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
args, unknown = parser.parse_known_args()
config = yaml.safe_load(open(args.config))
config = Config(**config)


class GenericDataset(Dataset):
    def __init__(self, filefolder, transform):

        assert filefolder is not None

        self.filefolder = filefolder
        self.transform = transform
        self.N_class = config.dataset.num_classes

    def __getitem__(self, index):

        filepath, label = self.filefolder[index]

        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if config.task.type == "classification":
            image = self.transform(image=image)["image"]
        elif config.task.type == "segmentation":
            label = np.asarray(Image.open(label))
            augmentations = self.transform(image=image, mask=label)
            image = augmentations.image
            label = augmentations.mask
            label = (
                torch.nn.functional.one_hot(label.to(torch.int64), self.N_class)
                .permute((2, 0, 1))
                .to(torch.float)
            )

        return image, label

    def __len__(self):
        return len(self.filefolder)


def generic_set_one_annotation(annotation_path, annotation_name, transformations_dict):

    assert annotation_path is not None

    annotation_path = os.path.join(annotation_path, annotation_name)

    if transformations_dict is None:
        train_transform = default_train_transform
        val_transform = default_val_transform
    else:
        train_transform = transformations_dict["train"]
        val_transform = transformations_dict["val"]

    trainfolder, valfolder = create_folders(annotation_path)

    trainset = GenericDataset(filefolder=trainfolder, transform=train_transform)

    valset = GenericDataset(filefolder=valfolder, transform=val_transform)

    return trainset, valset


def create_folders(annotation_path):
    trainfolder = []
    valfolder = []

    with open(annotation_path, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)
        source_path = os.path.dirname(annotation_path)

        for line in csvreader:

            imgpath = os.path.join(source_path, line[0])
            if config.task.type == "classification":
                class_id = int(line[1])
            elif config.task.type == "segmentation":
                class_id = os.path.join(source_path, line[1])

            if line[2] == "True" or line[2] == "1":
                valfolder.append([imgpath, class_id])
            else:
                trainfolder.append([imgpath, class_id])

    return trainfolder, valfolder


crop_shape = config.model.size
resize_shape = [int(crop_shape[0] * 1.1), int(crop_shape[1] * 1.1)]

# default_train_transform = transforms.Compose([
#             transforms.Resize(resize_shape),
#             transforms.RandomResizedCrop(crop_shape, scale=(0.25, 1)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

# default_val_transform = transforms.Compose([
#             transforms.Resize(resize_shape),
#             transforms.CenterCrop(crop_shape),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
default_val_transform = A.Compose(
    [
        A.Resize(resize_shape[1], resize_shape[0], p=1),
        A.CenterCrop(crop_shape[1], crop_shape[0], p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Ap.transforms.ToTensorV2(),
    ]
)
default_train_transform = A.Compose(
    [
        A.Resize(resize_shape[1], resize_shape[0], p=1),
        A.RandomResizedCrop(
            height=crop_shape[1], width=crop_shape[0], scale=(0.25, 1), p=1
        ),
        A.Flip(
            p=0.5
        ),  # Отразите вход по горизонтали, вертикали или по горизонтали и вертикали.
        A.Rotate(
            p=0.5
        ),  # Поверните ввод на угол, случайно выбранный из равномерного распределения.
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Ap.transforms.ToTensorV2(),
    ]
)
