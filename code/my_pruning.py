import subprocess


def potok(
    self_ind, cuda, load, name_sloi, sparsity, orig_size, iterr, algoritm, config_path
):
    #     print(self_ind, name_sloi)
    return subprocess.call(
        [
            "python3",
            "code/my_pruning_pabotnik.py",
            "--self_ind",
            str(self_ind),
            "--cuda",
            str(cuda),
            "--load",
            str(load),
            "--name_sloi",
            str(name_sloi),
            "--sparsity",
            str(sparsity),
            "--orig_size",
            str(orig_size),
            "--iterr",
            str(iterr),
            "--algoritm",
            str(algoritm),
            "--config",
            str(config_path),
        ],
    )


def my_pruning(start_size_model, config_path):
    import os
    import torch
    import torch.optim as optim
    import time
    import copy
    from multiprocessing import Process
    import json
    import yaml
    import pandas as pd
    import numpy as np
    import importlib
    from  ast import literal_eval

    from constants import Config
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    config = Config(**config)
    if config.training.is_self_traner:
        trainer = importlib.import_module(config.training.self_traner)
    else:
        trainer = importlib.import_module('training')

    # import training as trainer
    from my_pruning_pabotnik import get_size, get_stract, get_mask, get_top_list

    

    lr_training = config.training.lr
    N_it_ob = config.training.num_epochs

    load = config.my_pruning.restart.load
    load = load if load else os.path.join(
        config.path.exp_save, config.path.modelName, 'orig_model.pth'
    )
    start_iteration = config.my_pruning.restart.start_iteration
    alf = config.my_pruning.alf
    P = config.my_pruning.P
    cart = config.my_pruning.cart
    delta_crop = config.my_pruning.delta_crop
    resize_alf = config.my_pruning.resize_alf
    iskl = config.my_pruning.iskl
    algoritm = config.my_pruning.algoritm
    exp_save = config.path.exp_save
    modelName = config.path.modelName
    mask = config.mask.type
    sours_mask = config.mask.sours_mask
    class_name = config.class_name
    since = time.time()
    print(load)
    model = torch.load(load)
    top_n = config.my_pruning.top_n
    fil_importance = os.path.join(
        config.path.exp_save, config.path.modelName, f"{modelName}_importance.csv"
    )
    if not os.path.isfile(os.path.join(exp_save, f"{modelName}_log.csv")):
        f = open(os.path.join(exp_save, f"{modelName}_log.csv"), "w")
        f.write("N,sloi,do,posle,acc,size\n")
        f.close()
    # Кастыль для сегментации в офе
    if config.model.type_save_load == "interface":
        model.backbone_hooks._attach_hooks()

    if mask == "mask":
        if sours_mask == None:
            print(class_name)
            sours_mask = get_mask(model, class_name=class_name)
        else:
            with open(sours_mask, "r") as fp:
                sours_mask = json.load(fp)
        with open(os.path.join(config.path.exp_save, f"{modelName}.msk"), "w")  as fp:
            json.dump(sours_mask, fp)
    if not start_size_model:
        start_size_model = get_size(copy.deepcopy(model), config.model.size)
    if resize_alf:
        parametri = []
        ind = 0
        it = -1
        names = get_stract(model)
        for name in names:
            if name[1] == "Conv2d" and len(name[0].split(".bias")) == 1:
                add = True
                for isk in iskl:
                    if isk == name[0].split(".weight")[0]:
                        add = False
                if name[2][1] > alf and add and name[2][1] % alf != 0:
                    for name2 in get_stract(model):
                        if name[0] == name2[0] and name2[2][1] % alf != 0:
                            sprasity = (
                                name2[2][1] - (int(name2[2][1] / alf) * alf)
                            ) / name2[2][1]
                            parametri = [
                                ind,
                                cart[0],
                                load,
                                name[0].split(".weight")[0],
                                sprasity,
                                name2[2][0] * name2[2][1],
                                it,
                                algoritm,
                                config_path,
                            ]
                            print(parametri)
                            log = open(
                                os.path.join(config.path.exp_save, "log.txt"), "a"
                            )
                            log.write(str(parametri) + "\n")
                            log.close()
                            ind += 1
                            p = Process(target=potok, args=parametri)
                            p.start()
                            p.join()
                            fil_it = os.path.join(
                                config.path.exp_save,
                                config.path.modelName,
                                f"{modelName}_it{it}.csv",
                            )
                            f = open(fil_it, "r")
                            strr = f.read().split("\n")
                            if strr[-2].split(" ")[2] != "EROR":
                                maxx = float(strr[-2].split(" ")[2])
                                load = os.path.join(
                                    config.path.exp_save,
                                    config.path.modelName,
                                    f"{modelName}_{strr[-2].split(' ')[1]}_it_{it}_acc_{maxx:.3f}.pth",
                                )
                            print(strr[-2])
                            log = open(
                                os.path.join(config.path.exp_save, "log.txt"), "a"
                            )
                            log.write(str(strr[-2]) + "\n")
                            log.close()
                            model = torch.load(load)
                            # Кастыль для сегментации в офе
                            if config.model.type_save_load == "interface":
                                model.backbone_hooks._attach_hooks()
        if len(parametri):
            size_model = get_size(copy.deepcopy(model), config.model.size)
            optimizer = optim.Adam(model.parameters(), lr=lr_training)
            model, loss, acc, st, time_elapsed2 = trainer.trainer(
                model, optimizer, trainer.criterion, num_epochs=N_it_ob, ind=f"it{it}"
            )
            load = os.path.join(
                config.path.exp_save,
                config.path.modelName,
                f"{modelName}_it_{it}_acc_{acc:.3f}_size_{size_model / start_size_model:.3f}.pth",
            )
            # Кастыль для сегментации в офе
            if config.model.type_save_load == "interface":
                model.backbone_hooks._clear_hooks()
            torch.save(model, load)
            f = open(os.path.join(exp_save, f"{modelName}_log.csv"), "a")
            f.write(f"{it},,,,{acc},{size_model / start_size_model}\n")
            f.close()
            for filename in os.listdir(
                os.path.join(config.path.exp_save, config.path.modelName)
            ):
                if filename.split(".")[-1] == "pth":
                    if (
                        len(filename.split("size")) == 1
                        and filename != "orig_model.pth"
                    ):
                        os.remove(
                            os.path.join(
                                config.path.exp_save, config.path.modelName, filename
                            )
                        )
                if filename.split(".")[-1] == "txt":
                    if len(filename.split("train_log")) == 2:
                        os.remove(
                            os.path.join(
                                config.path.exp_save, config.path.modelName, filename
                            )
                        )

    it = start_iteration
    size_model = get_size(model, config.model.size)
    importance = pd.DataFrame()
    next = True
    ind = 0

    top_list = []
    if os.path.isfile(fil_importance):
        importance = pd.read_csv(fil_importance)
        top_list = get_top_list(importance.values, iskl, model, top_n, alf, i_start=0, is_literal_eval = True)
    else:
        temp_list = get_stract(model)
        top_list = get_top_list(temp_list, iskl, model, len(temp_list), alf, i_start=0, is_literal_eval = False)
    
    while start_size_model * (1 - P) < size_model:
        print(start_size_model, size_model, start_size_model * (1 - P))
        log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
        log.write(f"{start_size_model}, {size_model}, {start_size_model * (1 - P)}\n")
        log.close()
        parametri = []
        model = torch.load(load)
        for name in top_list:
            submodule = model.get_submodule(name[0])
            size = [submodule.in_channels, submodule.out_channels]
            a = int(size[1] // alf * delta_crop)
            if size[1] >= alf * 2 and a == 0:
                a = a + 1
            b = size[1] % alf
            sprasity = (a * alf + b) / size[1]
            parametri.append(
                (
                    ind,
                    cart[ind % len(cart)],
                    load,
                    name[0],
                    sprasity,
                    size[0] * size[1],
                    it,
                    algoritm,
                    config_path,
                )
            )
            ind += 1
                    # elif submodule.in_channels == submodule.out_channels:
                    #     parametri.append(
                    #         (
                    #             ind,
                    #             cart[ind % len(cart)],
                    #             load,
                    #             name[0],
                    #             1,
                    #             name[2][0] * name[2][1],
                    #             it,
                    #             'delete',
                    #             config_path,
                    #         )
                    #     )
                    #     ind += 1
                # elif (isinstance(submodule, torch.nn.Conv2d) 
                #       and submodule.in_channels == submodule.groups 
                #       and submodule.in_channels == alf):
                #     parametri.append(
                #         (
                #             ind,
                #             cart[ind % len(cart)],
                #             load,
                #             name[0],
                #             1,
                #             name[2][0] * name[2][1],
                #             it,
                #             'delete',
                #             config_path,
                #         )
                #     )
                #     ind += 1
                # elif (isinstance(submodule, torch.nn.BatchNorm2d) 
                #       and submodule.num_features == alf):
                #     parametri.append(
                #         (
                #             ind,
                #             cart[ind % len(cart)],
                #             load,
                #             name[0],
                #             1,
                #             name[2][0],
                #             it,
                #             'delete',
                #             config_path,
                #         )
                #     )
                #     ind += 1
                # elif (isinstance(submodule, torch.nn.Linear) 
                #       and submodule.in_features == submodule.out_features == alf):
                #     parametri.append(
                #         (
                #             ind,
                #             cart[ind % len(cart)],
                #             load,
                #             name[0],
                #             1,
                #             name[2][0] * name[2][1],
                #             it,
                #             'delete',
                #             config_path,
                #         )
                #     )
                #     ind += 1
        del model
        opit = 0
        since3 = time.time()
        while opit < len(parametri):
            since2 = time.time()
            rab = []
            for j in cart:
                if opit < len(parametri):
                    print(parametri[opit])
                    log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
                    log.write(f"{parametri[opit]}\n")
                    log.close()
                    p = Process(target=potok, args=parametri[opit])
                    p.start()
                    rab.append(p)
                    opit += 1
            for p in rab:
                p.join()
            model = torch.load(load)
            time_elapsed = time.time() - since2
            print(f"time_iter= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
            log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
            log.write(
                f"time_iter= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n"
            )
            log.close()

        fil_it = os.path.join(
            config.path.exp_save, config.path.modelName, f"{modelName}_it{it}.csv"
        )
        data = pd.read_csv(fil_it)
        
        if not os.path.isfile(fil_importance):
            importance = data.sort_values('acc', ascending = False)[['name', 'acc', 'before', 'after']]
            top_list = get_top_list(importance.values, iskl, model, top_n, alf, i_start=0, is_literal_eval = True)
        else:
            importance_t = data.sort_values('acc', ascending = False)[['name', 'acc', 'before', 'after']]
            top_list = get_top_list(importance.values, iskl, model, top_n, alf, i_start=len(importance_t), is_literal_eval = True)
            if len(top_list)==0 or top_list[0][1] < importance_t.values[0][1]:
                next = True
            else:
                next = False
                del model
        if next:
            for i, d in data[['name', 'acc', 'before', 'after']].iterrows():
                ind = importance[(importance['name'] == d['name'])]
                if len(ind):
                    importance.loc[ind.index,'acc'] = d['acc']
                    importance.loc[ind.index,'before'] = d['before']
                    importance.loc[ind.index,'after'] = d['after']
                else:
                    importance.loc[len(importance)] = d
            importance = importance.sort_values('acc', ascending = False)
            top_list = get_top_list(importance.values, iskl, model, top_n, alf, i_start=0, is_literal_eval = True)
            ind = 0
            fil_importance = os.path.join(
                config.path.exp_save, config.path.modelName, f"{modelName}_importance.csv"
            )
            importance.to_csv(fil_importance, index=False)
            importance.to_csv(os.path.join(
                config.path.exp_save, config.path.modelName, f"{modelName}_importance_it{it}.csv"
            ), index=False)

            if not np.isnan(data["acc"].max()):
                i = data["acc"].idxmax()
                _, sloi, acc, before, after = data.iloc[[i]].values[0]
                load = os.path.join(
                    config.path.exp_save,
                    config.path.modelName,
                    f"{modelName}_{sloi}_it_{it}_acc_{acc:.3f}.pth",
                )
            else:
                break
            model = torch.load(load)
            m = copy.deepcopy(model)
            # Кастыль для сегментации в офе
            if config.model.type_save_load == "interface":
                model.backbone_hooks._attach_hooks()
                m.backbone_hooks._attach_hooks()

            optimizer = optim.Adam(model.parameters(), lr=lr_training)
            size_model = get_size(m, config.model.size)
            del m
            model, loss, acc, st, time_elapsed2 = trainer.trainer(
                model, optimizer, trainer.criterion, num_epochs=N_it_ob, ind=f"it{it}"
            )
            load = os.path.join(
                config.path.exp_save,
                config.path.modelName,
                f"{modelName}_it_{it}_acc_{acc:.3f}_size_{size_model / start_size_model:.3f}.pth",
            )
            # Кастыль для сегментации в офе
            if config.model.type_save_load == "interface":
                model.backbone_hooks._clear_hooks()
            torch.save(model, load)
            del model
            f = open(os.path.join(exp_save, f"{modelName}_log.csv"), "a")
            f.write(
                f'{it},{sloi},"{before}","{after}",{acc},{size_model / start_size_model}\n'
            )
            f.close()
            for filename in os.listdir(
                os.path.join(config.path.exp_save, config.path.modelName)
            ):
                if filename.split(".")[-1] == "pth":
                    if len(filename.split("size")) == 1 and filename != "orig_model.pth":
                        os.remove(
                            os.path.join(
                                config.path.exp_save, config.path.modelName, filename
                            )
                        )
                if filename.split(".")[-1] == "txt":
                    if len(filename.split("train_log")) == 2:
                        os.remove(
                            os.path.join(
                                config.path.exp_save, config.path.modelName, filename
                            )
                        )
            it = it + 1
            time_elapsed = time.time() - since3
            print(f"time_epox= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
            log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
            log.write(f"time_epox= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n")
            log.close()

            # # Обновление маски после удаления параметров
            # log = pd.read_csv(os.path.join(exp_save, f"{modelName}_log.csv"))
            # if log["posle"].isnull().iloc[-1]:
            #     sours_mask = get_mask(model, class_name=class_name)
            #     with open(os.path.join(config.path.exp_save, f"{modelName}.msk"), "w")  as fp:
            #         json.dump(sours_mask, fp)

    time_elapsed = time.time() - since
    print(f"time_total= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(start_size_model, size_model, start_size_model * (1 - P))
    log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
    log.write(f"time_total= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n")
    log.write(f"{start_size_model}, {size_model}, {start_size_model * (1 - P)}\n")
    log.close()
