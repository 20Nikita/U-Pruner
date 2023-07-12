import os
import yaml
config = yaml.safe_load(open('Pruning.yaml'))
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config['model']['gpu'])

import torch
import copy
import time
import torch.optim as optim

import nni
from nni.algorithms.compression.v2.pytorch.pruning import L2NormPruner, FPGMPruner, TaylorFOWeightPruner, AGPPruner, LinearPruner, LotteryTicketPruner

import cod.my_pruning as my_pruning
from cod.my_pruning_pabotnik import get_size, get_stract, rename
from cod.ModelSpeedup import ModelSpeedup
import cod.training as trainer

since = time.time()
acc = 0
component = 'pruning'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

snp = config['path']['exp_save'] + "/" + config['path']['model_name']
if not os.path.exists(snp): os.makedirs(snp)

type_save_load = config['model']['type_save_load']
if type_save_load == 'interface':
    from interfaces.tools import *
    interface = load_interface(path_to_interface = config['model']['path_to_resurs'], load_name = config['model']['name_resurs'])
    model = build_net(interface=interface, pretrained = True)
    # model = build_net(interface=interface, pretrained = False)
elif type_save_load == 'pth':
    import sys
    sys.path.append(config['model']['path_to_resurs'])
    model = torch.load(f"{config['model']['path_to_resurs']}/{config['model']['name_resurs']}.pth")
    
print(config['algorithm'])

if config['algorithm'] == 'My_pruning':
    # Кастыль для сегментации в офе
    if type_save_load == 'interface' and config['task']['type'] == "segmentation":
        model.backbone_hooks._clear_hooks()
    torch.save(model, snp + "/" + "orig_model.pth")
    # Кастыль для сегментации в офе
    if type_save_load == 'interface' and config['task']['type'] == "segmentation":
        model.backbone_hooks._attach_hooks()
    start_size_model = get_size(model)
    del model
    my_pruning.my_pruning(start_size_model = start_size_model)
    _, N, _, sloi, _, _, _, _, _, _, _, acc, _, size = open(config['path']['exp_save']+"/"+config['path']['model_name']+"_log.txt").readlines()[-1].split(" ")
    model = torch.load(f"{config['path']['exp_save']}/{config['path']['model_name']}/{config['path']['model_name']}_it_{N}_acc_{float(acc):.3f}_size_{float(size):.3f}.pth")
    
else:
    model = model.to(device)
    traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
    config_list = [{ 'sparsity': config['nni_pruning']['P'], 'op_types': ['Conv2d'] }]
    pruning_params = {
        'trainer': trainer.retrainer,
        'traced_optimizer': traced_optimizer,
        'criterion': trainer.criterion,
        'training_batches': config['dataset']['dataLoader']['batch_size_t']
    }
    
    if config['algorithm'] == 'L2Norm':
        pruner = L2NormPruner(model, config_list)
    elif  config['algorithm'] == 'FPGM':
        pruner = FPGMPruner(model, config_list)
    elif  config['algorithm'] == 'TaylorFOWeight':
        pruner = TaylorFOWeightPruner(model, config_list, trainer.retrainer, traced_optimizer, trainer.criterion, training_batches=config['dataset']['dataLoader']['batch_size_t'])
    elif  config['algorithm'] == 'AGP':
        pruner = AGPPruner(model, config_list, pruning_algorithm='taylorfo', total_iteration=config['nni_pruning']['total_iteration'], finetuner=trainer.finetuner, log_dir = snp, pruning_params = pruning_params)
    elif  config['algorithm'] == 'Linear':
        pruner = LinearPruner(model, config_list, pruning_algorithm='taylorfo', total_iteration=config['nni_pruning']['total_iteration'], finetuner=trainer.finetuner, log_dir = snp, pruning_params = pruning_params)
    elif  config['algorithm'] == 'LotteryTicket':
        pruner = LotteryTicketPruner(model, config_list, pruning_algorithm='taylorfo', total_iteration=config['nni_pruning']['total_iteration'], finetuner=trainer.finetuner, log_dir = snp, pruning_params = pruning_params)
    if config['algorithm'] == 'L2Norm' or config['algorithm'] == 'FPGM' or config['algorithm'] == 'TaylorFOWeight':
        masked_model, masks = pruner.compress()
    else:
        pruner.compress()
        _, model, masks, _, _ = pruner.get_best_result()
    pruner._unwrap_model()
    model = ModelSpeedup(model, masks).to(device)
    if config['nni_pruning']['training']:
        optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
        model, loss, acc, st, time_elapsed2 = trainer.trainer(model, optimizer, trainer.criterion, num_epochs = config['training']['num_epochs'])    

if type_save_load == 'interface':         
    model_orig = build_net(interface=interface)
    model_prun = model
    stract1 = get_stract(model_orig)
    stract2 = get_stract(model_prun)

    m = copy.deepcopy(model_prun)
    mo = copy.deepcopy(model_orig)
    if type_save_load == 'interface' and config['task']['type'] == "segmentation":
        m.backbone_hooks._attach_hooks()
        mo.backbone_hooks._attach_hooks()
    size = get_size(m) / get_size(mo)
    params = model_prun.state_dict()
    if ('pruning' in interface) and len(interface['pruning']['summary']['resize']):
        resize = interface['pruning']['summary']['resize']
    else:
        resize = []
        interface['pruning'] = {}
        interface['pruning']['summary'] = {}
        interface['pruning']['is_pruning'] = True
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

    save_interface(params=params, interface=interface, name_component=component, summary=summary,path_to_interface = config['model']['path_to_resurs'])
elif  type_save_load == 'pth':
    torch.save(model, f"{config['model']['path_to_resurs']}/rezalt.pth")