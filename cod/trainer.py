import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cod.dataset as Dataset
from cod.train_nni import train_model

config = yaml.safe_load(open('Pruning.yaml'))

N_class = config['dataset']['num_classes']
annotation_path = config['dataset']['annotation_path']

classification_criterion = nn.CrossEntropyLoss()
criterion = classification_criterion

batch_size_t  = config['dataset']['dataLoader']['batch_size_t']
num_workers_t = config['dataset']['dataLoader']['num_workers_t']
pin_memory_t  = config['dataset']['dataLoader']['pin_memory_t']
drop_last_t   = config['dataset']['dataLoader']['drop_last_t']
shuffle_t     = config['dataset']['dataLoader']['shuffle_t']
batch_size_v  = config['dataset']['dataLoader']['batch_size_v']
num_workers_v = config['dataset']['dataLoader']['num_workers_v']
pin_memory_v  = config['dataset']['dataLoader']['pin_memory_v']
drop_last_v   = config['dataset']['dataLoader']['drop_last_v']
shuffle_v     = config['dataset']['dataLoader']['shuffle_v']

trainset, valset = Dataset.generic_set_one_annotation(annotation_path, None)
train_dataloader = DataLoader(trainset, batch_size=batch_size_t, num_workers=num_workers_t, pin_memory=pin_memory_t, drop_last=drop_last_t, shuffle=shuffle_t)
val_dataloader = DataLoader(valset, batch_size=batch_size_v, num_workers=num_workers_v, pin_memory=pin_memory_v, drop_last=drop_last_v, shuffle=shuffle_v)


def trainer(model, optimizer, criterion, epoch = 0, num_epochs = 1, ind = 0):
    fil = config['path']['exp_save'] + "/" + config['path']['model_name'] + "/" + f"train_log_{ind}.txt"
    model, _, _, pihati, mass, time_elapsed  = train_model(model, 
                                    criterion,
                                    optimizer, 
                                    train_dataloader, 
                                    val_dataloader, 
                                    batch_size_t, 
                                    batch_size_v, 
                                    num_epochs = num_epochs,
                                    N_class = N_class,
                                    rezim = ['T','V'],
                                    fil = fil)
    mi = mass[2].index(min(mass[2]))
    ma = mass[1].index(max(mass[1]))
    st = pihati
    return model, mass[2][mi], mass[1][ma], st, time_elapsed



