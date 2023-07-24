import torch
import copy
from cod.my_pruning_pabotnik import  get_stract, compres2

def pruning_type(model, masks, type_pruning = "defolt"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type_pruning == "defolt":
        compres2(model, masks)
    elif type_pruning == "ofa":
        compres2(model, masks, ofa = True)
    elif type_pruning == "total":
        compres2(model, masks, ofa = True, stop = False)
    model.to(device)
    model(torch.rand(1,3,640,480).to(device))
    return model

def ModelSpeedup(model, masks):
    stract =  get_stract(model)
    nev_masks = [{mask:masks[mask]} for mask in masks]

    for i in nev_masks:
        
        orig_size = tek_size = 0
        for name in stract:
            if name[0].split(".weight")[0].split(".bias")[0] == list(i.keys())[0]:
                orig_size = name[2]
        for name in get_stract(model):
            if name[0].split(".weight")[0].split(".bias")[0] == list(i.keys())[0]:
                tek_size = name[2]
        
        if tek_size == orig_size:
            try:
                model = pruning_type(copy.deepcopy(model), i, type_pruning = "defolt")
            except:
                try:
                    model = pruning_type(copy.deepcopy(model), i, type_pruning = "ofa")
                except:
                    model = pruning_type(copy.deepcopy(model), i, type_pruning = "total")
                    
    return model