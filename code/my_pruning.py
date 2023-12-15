import subprocess


def potok(
    self_ind, cuda, load, name_sloi, sparsity, orig_size, iterr, algoritm, config_path
):
    #     print(self_ind, name_sloi)
    return subprocess.call(
        [
            "python",
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

    import training as trainer
    from my_pruning_pabotnik import get_size, get_stract, get_mask
    from constants import Config

    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    config = Config(**config)

    lr_training = config.training.lr
    N_it_ob = config.training.num_epochs

    load = config.my_pruning.restart.load
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
    model = torch.load(load)
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
        with open(load.split(".")[0] + ".msk", "w") as fp:
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
    stract = get_stract(model)
    size_model = get_size(model, config.model.size)
    del model
    while start_size_model * (1 - P) < size_model:
        print(start_size_model, size_model, start_size_model * (1 - P))
        log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
        log.write(f"{start_size_model}, {size_model}, {start_size_model * (1 - P)}\n")
        log.close()
        parametri = []
        ind = 0
        for name in stract:
            if name[1] == "Conv2d" and len(name[0].split(".bias")) == 1:
                add = True
                for isk in iskl:
                    if isk == name[0].split(".weight")[0]:
                        add = False
                if name[2][1] > alf and add:
                    sprasity = 0
                    if delta_crop == None:
                        sprasity = (
                            name[2][1] - (int(name[2][1] * (1 - P) / alf) * alf)
                        ) / name[2][1]
                    else:
                        sprasity = (
                            name[2][1]
                            - (int(name[2][1] * (1 - delta_crop) / alf) * alf)
                        ) / name[2][1]
                    if sprasity == 1:
                        sprasity = (
                            name[2][1] - ((int(name[2][1] / alf) - 1) * alf)
                        ) / name[2][1]
                        if sprasity == 1:
                            sprasity = (
                                name[2][1] - (int(name[2][1] / alf) * alf)
                            ) / name[2][1]
                    parametri.append(
                        (
                            ind,
                            cart[ind % len(cart)],
                            load,
                            name[0].split(".weight")[0],
                            sprasity,
                            name[2][0] * name[2][1],
                            it,
                            algoritm,
                            config_path,
                        )
                    )
                    ind += 1
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
        if it == start_iteration:
            i = np.where(pd.isnull(data))[0]
            arr = data["name"].values[i]
            iskl = np.concatenate((iskl, arr))
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
        stract = get_stract(model)
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
    time_elapsed = time.time() - since
    print(f"time_total= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(start_size_model, size_model, start_size_model * (1 - P))
    log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
    log.write(f"time_total= {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n")
    log.write(f"{start_size_model}, {size_model}, {start_size_model * (1 - P)}\n")
    log.close()
