import yaml
import torch
import copy
import time
import torch.optim as optim

from nni.compression.pytorch.pruning import L2NormPruner

from interfaces.tools import *
import cod.my_pruning as my_pruning
from cod.my_pruning_pabotnik import get_size, get_stract, rename
from cod.ModelSpeedup import ModelSpeedup
import cod.training as trainer

since = time.time()
acc = 0
component = 'pruning'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = yaml.safe_load(open('Pruning.yaml'))

snp = config['path']['exp_save'] + "/" + config['path']['model_name']
if not os.path.exists(snp): os.makedirs(snp)

interface = load_interface(path_to_interface = config['model']['path_to_interface'])
model = build_net(interface=interface, pretrained = True)

print(config['algorithm'])

if config['algorithm'] == 'My_pruning':
    torch.save(model, snp + "/" + "orig_model.pth")
    start_size_model = get_size(model)
    del model
    my_pruning.my_pruning(start_size_model = start_size_model)
    _, N, _, sloi, _, _, _, _, _, _, _, acc, _, size = open(config['path']['exp_save']+"/"+config['path']['model_name']+"_log.txt").readlines()[-1].split(" ")
    model = torch.load(f"{config['path']['exp_save']}/{config['path']['model_name']}/{config['path']['model_name']}_it_{N}_acc_{float(acc):.3f}_size_{float(size):.3f}.pth")
    
elif  config['algorithm'] == 'L2Norm':
    config_list = [{ 'sparsity': config['L2Norm']['P'], 'op_types': ['Conv2d'] }]
    pruner = L2NormPruner(model, config_list)
    masked_model, masks = pruner.compress()
    pruner._unwrap_model()
    model = ModelSpeedup(model, masks).to(device)
    if config['L2Norm']['training']:
        optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
        model, loss, acc, st, time_elapsed2 = trainer.trainer(model, optimizer, trainer.criterion, num_epochs = config['training']['num_epochs'])
        
    
model_orig = build_net(interface=interface)
model_prun = model
stract1 = get_stract(model_orig)
stract2 = get_stract(model_prun)

size = get_size(copy.deepcopy(model_prun)) / get_size(copy.deepcopy(model_orig))
params = model_prun.state_dict()
resize = []
for i in range(len(stract1)):
    if stract1[i][2]!=stract2[i][2]:
        resize.append({ "name": rename(stract1[i][0]), "type": stract1[i][1], "orig_shape": stract1[i][2],"shape": stract2[i][2]})
        
time_elapsed = time.time() - since
summary = {
        'size': float(size),
        'val_accuracy': float(acc),
        'resize': resize,
        'time': "{:.0f}h {:.0f}m {:.0f}s".format(time_elapsed // 60 // 60, time_elapsed // 60 - time_elapsed // 60 // 60 * 60, time_elapsed % 60)
}
interface['pruning'] = {}
interface['pruning']['summary'] = {}
interface['pruning']['is_pruning'] = True
save_interface(interface=interface, name_component=component, params=params, summary=summary)