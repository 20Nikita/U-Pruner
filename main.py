import yaml
import torch
import copy

from interfaces.tools import *
import my_pruning
from my_pruning_pabotnik import get_size

config = yaml.safe_load(open('Pruning.yaml'))

snp = config['path']['exp_save'] + "/" + config['path']['model_name']
if not os.path.exists(snp): os.makedirs(snp)

interface = load_interface(path_to_interface = config['model']['path_to_interface'])
model = build_net(interface=interface, pretrained = True)


torch.save(model, snp + "/" + "orig_model.pth")
start_size_model = get_size(model)
del model

my_pruning.my_pruning(start_size_model = start_size_model)