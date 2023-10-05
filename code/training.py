import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import dataset as Dataset
import detect
from train_nni import train_model
from constants import DEFAULT_CONFIG_PATH, Config
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH)

args, unknown = parser.parse_known_args()
config = yaml.safe_load(open(args.config))
config = Config(**config)

N_class = config.dataset.num_classes
size = config.model.size
annotation_path = config.dataset.annotation_path
annotation_name = config.dataset.annotation_name

segmentation_criterion = nn.BCEWithLogitsLoss()
classification_criterion = nn.CrossEntropyLoss()
detection_criterion = detect.Loss

batch_size_t = config.retraining.dataLoader.batch_size_t
num_workers_t = config.retraining.dataLoader.num_workers_t
pin_memory_t = config.retraining.dataLoader.pin_memory_t
drop_last_t = config.retraining.dataLoader.drop_last_t
shuffle_t = config.retraining.dataLoader.shuffle_t
batch_size_v = config.retraining.dataLoader.batch_size_v
num_workers_v = config.retraining.dataLoader.num_workers_v
pin_memory_v = config.retraining.dataLoader.pin_memory_v
drop_last_v = config.retraining.dataLoader.drop_last_v
shuffle_v = config.retraining.dataLoader.shuffle_v

tbatch_size_t = config.training.dataLoader.batch_size_t
tnum_workers_t = config.training.dataLoader.num_workers_t
tpin_memory_t = config.training.dataLoader.pin_memory_t
tdrop_last_t = config.training.dataLoader.drop_last_t
tshuffle_t = config.training.dataLoader.shuffle_t
tbatch_size_v = config.training.dataLoader.batch_size_v
tnum_workers_v = config.training.dataLoader.num_workers_v
tpin_memory_v = config.training.dataLoader.pin_memory_v
tdrop_last_v = config.training.dataLoader.drop_last_v
tshuffle_v = config.training.dataLoader.shuffle_v

task_tupe = config.task.type
if annotation_name == None:
    annotation_train = config.dataset.annotation_name_train
    annotation_val = config.dataset.annotation_name_val

ssd = False
if config.task.type == "detection" and config.task.detection == "ssd":
    ssd = True

if task_tupe == "classification":
    criterion = classification_criterion
elif task_tupe == "segmentation":
    criterion = segmentation_criterion
elif task_tupe == "detection":
    criterion = detection_criterion

if task_tupe != "detection":
    trainset, valset = Dataset.generic_set_one_annotation(
        annotation_path, annotation_name, None
    )
    train_dataloader = DataLoader(
        trainset,
        batch_size=batch_size_t,
        num_workers=num_workers_t,
        pin_memory=pin_memory_t,
        drop_last=drop_last_t,
        shuffle=shuffle_t,
    )
    val_dataloader = DataLoader(
        valset,
        batch_size=batch_size_v,
        num_workers=num_workers_v,
        pin_memory=pin_memory_v,
        drop_last=drop_last_v,
        shuffle=shuffle_v,
    )
    ttrain_dataloader = DataLoader(
        trainset,
        batch_size=tbatch_size_t,
        num_workers=tnum_workers_t,
        pin_memory=tpin_memory_t,
        drop_last=tdrop_last_t,
        shuffle=tshuffle_t,
    )
    tval_dataloader = DataLoader(
        valset,
        batch_size=tbatch_size_v,
        num_workers=tnum_workers_v,
        pin_memory=tpin_memory_v,
        drop_last=tdrop_last_v,
        shuffle=tshuffle_v,
    )
else:
    train_transform, val_transform = detect.get_ransforms(size[0], size[1])
    val_dataset = detect.MyDataset(
        os.path.join(annotation_path, annotation_val),
        annotation_path,
        "val",
        val_transform,
    )
    train_dataset = detect.MyDataset(
        os.path.join(annotation_path, annotation_train),
        annotation_path,
        "train",
        train_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_t,
        num_workers=num_workers_t,
        pin_memory=pin_memory_t,
        drop_last=drop_last_t,
        shuffle=shuffle_t,
        collate_fn=detect.custom_collate,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_v,
        num_workers=num_workers_v,
        pin_memory=pin_memory_v,
        drop_last=drop_last_v,
        shuffle=shuffle_v,
        collate_fn=detect.custom_collate,
    )
    ttrain_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=tbatch_size_t,
        num_workers=tnum_workers_t,
        pin_memory=tpin_memory_t,
        drop_last=tdrop_last_t,
        shuffle=tshuffle_t,
        collate_fn=detect.custom_collate,
    )
    tval_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=tbatch_size_v,
        num_workers=tnum_workers_v,
        pin_memory=tpin_memory_v,
        drop_last=tdrop_last_v,
        shuffle=tshuffle_v,
        collate_fn=detect.custom_collate,
    )


def retrainer(model, optimizer, criterion, epoch=0, num_epochs=1, ind=0):
    fil = os.path.join(
        config.path.exp_save, config.path.modelName, f"train_log_{ind}.txt"
    )

    model, _, _, pihati, mass, time_elapsed = train_model(
        model,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        batch_size_t,
        batch_size_v,
        num_epochs=num_epochs,
        N_class=N_class,
        rezim=["T", "V"],
        fil=fil,
        task_tupe=task_tupe,
        ssd=ssd,
    )
    mi = mass[2].index(min(mass[2]))
    ma = mass[1].index(max(mass[1]))
    st = pihati
    return model, mass[2][mi], mass[1][ma], st, time_elapsed


def trainer(model, optimizer, criterion, epoch=0, num_epochs=1, ind=0):
    fil = os.path.join(
        config.path.exp_save, config.path.modelName, f"train_log_{ind}.txt"
    )
    model, _, _, pihati, mass, time_elapsed = train_model(
        model,
        criterion,
        optimizer,
        ttrain_dataloader,
        tval_dataloader,
        tbatch_size_t,
        tbatch_size_v,
        num_epochs=num_epochs,
        N_class=N_class,
        rezim=["T", "V"],
        fil=fil,
        task_tupe=task_tupe,
        ssd=ssd,
    )
    mi = mass[2].index(min(mass[2]))
    ma = mass[1].index(max(mass[1]))
    st = pihati
    return model, mass[2][mi], mass[1][ma], st, time_elapsed


def finetuner(model):
    optimizer = optim.Adam(model.parameters(), lr=config.retraining.lr)
    fil = os.path.join(config.path.exp_save, config.path.modelName, "train_log_0.txt")
    train_model(
        model,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        batch_size_t,
        batch_size_v,
        num_epochs=config.retraining.num_epochs,
        N_class=N_class,
        rezim=["T", "V"],
        fil=fil,
        task_tupe=task_tupe,
        ssd=ssd,
    )
