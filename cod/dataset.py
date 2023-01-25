from torch.utils.data import Dataset
from PIL import Image
import os
import csv
import yaml

from torchvision import transforms

config = yaml.safe_load(open('/workspace/proj/Pruning.yaml'))

class GenericDataset(Dataset):
    def __init__(self, filefolder, transform):

        assert filefolder is not None

        self.filefolder = filefolder
        self.transform = transform

    def __getitem__(self, index):

        filepath, label = self.filefolder[index]

        image = Image.open(filepath)
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filefolder)


def generic_set_one_annotation(annotation_path, transformations_dict):

    assert annotation_path is not None

    annotation_path= os.path.join(annotation_path, "data.csv")

    if transformations_dict is None:
        train_transform = default_train_transform
        val_transform = default_val_transform
    else:
        train_transform = transformations_dict["train"]
        val_transform = transformations_dict["val"]

    trainfolder, valfolder = create_folders(
        annotation_path
    )

    trainset = GenericDataset(
        filefolder=trainfolder,
        transform=train_transform
    )

    valset = GenericDataset(
        filefolder=valfolder,
        transform=val_transform
    )

    return trainset, valset


def create_folders(annotation_path):
    trainfolder = []
    valfolder = []

    with open(annotation_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)
        source_path = os.path.dirname(annotation_path)

        for line in csvreader:

            imgpath = os.path.join(source_path, line[0])
            class_id = int(line[1])

            if line[2] == "True":
                valfolder.append(
                    [imgpath,
                     class_id]
                )
            else:
                trainfolder.append(
                    [imgpath,
                     class_id]
                )

    return trainfolder, valfolder


crop_shape = config['model']['size']
resize_shape = [int(crop_shape[0]*1.1),int(crop_shape[1]*1.1)]

default_train_transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.RandomResizedCrop(crop_shape, scale=(0.25, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

default_val_transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.CenterCrop(crop_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
