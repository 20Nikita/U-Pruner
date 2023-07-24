import subprocess
from collections import OrderedDict


def potok(self_ind, cuda, load, name_sloi, sparsity, orig_size, iterr, algoritm):
    #     print(self_ind, name_sloi)
    return subprocess.call(
        [
            "python3 cod/my_pruning_pabotnik.py --self_ind {}\
    --cuda {} --load {} --name_sloi {} --sparsity {} --orig_size {} --iterr {} --algoritm {}\
    ".format(
                self_ind, cuda, load, name_sloi, sparsity, orig_size, iterr, algoritm
            )
        ],
        shell=True,
    )


def my_pruning(start_size_model):
    import os
    import yaml

    config = yaml.safe_load(open("Pruning.yaml"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config["my_pruning"]["cart"][0])
    import torch
    import torch.optim as optim
    import time
    import copy
    from multiprocessing import Process
    import json

    import cod.training as trainer
    from cod.my_pruning_pabotnik import get_size, get_stract, rename, get_mask

    lr_training = config["training"]["lr"]
    N_it_ob = config["training"]["num_epochs"]

    load = config["my_pruning"]["restart"]["load"]
    start_iteration = config["my_pruning"]["restart"]["start_iteration"]
    alf = config["my_pruning"]["alf"]
    P = config["my_pruning"]["P"]
    cart = config["my_pruning"]["cart"]
    delta_crop = config["my_pruning"]["delta_crop"]
    resize_alf = config["my_pruning"]["resize_alf"]
    iskl = config["my_pruning"]["iskl"]
    algoritm = config["my_pruning"]["algoritm"]
    snp = config["path"]["exp_save"] + "/" + config["path"]["model_name"]
    exp_save = config["path"]["exp_save"]
    modelName = config["path"]["model_name"]
    mask = config["mask"]["type"]
    sours_mask = config["mask"]["sours_mask"]

    since = time.time()
    model = torch.load(load)
    # Кастыль для сегментации в офе
    if (
        config["model"]["type_save_load"] == "interface"
        and config["task"]["type"] == "segmentation"
    ):
        model.backbone_hooks._attach_hooks()

    if mask == "mask":
        if sours_mask == "None":
            sours_mask = get_mask(model)
        else:
            with open(sours_mask, "r") as fp:
                sours_mask = json.load(fp)
        with open(load.split(".")[0] + ".msk", "w") as fp:
            json.dump(sours_mask, fp)

    if not start_size_model:
        start_size_model = get_size(copy.deepcopy(model))
    if resize_alf:
        parametri = []
        ind = 0
        it = -1
        names = get_stract(model)
        for name in names:
            if (
                name[1] == "torch.nn.modules.conv.Conv2d"
                and len(name[0].split(".bias")) == 1
            ):
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
                            ]
                            print(parametri)
                            ind += 1
                            p = Process(target=potok, args=parametri)
                            p.start()
                            p.join()
                            fil_it = snp + "/" + modelName + "_it{}.txt".format(it)
                            f = open(fil_it, "r")
                            strr = f.read().split("\n")
                            if strr[-2].split(" ")[2] != "EROR":
                                maxx = float(strr[-2].split(" ")[2])
                                load = (
                                    snp
                                    + "/"
                                    + modelName
                                    + "_"
                                    + strr[-2].split(" ")[1]
                                    + "_it_{}_acc_{:.3f}.pth".format(it, maxx)
                                )
                            print(strr[-2])
                            model = torch.load(load)
                            # Кастыль для сегментации в офе
                            if (
                                config["model"]["type_save_load"] == "interface"
                                and config["task"]["type"] == "segmentation"
                            ):
                                model.backbone_hooks._attach_hooks()
        if len(parametri):
            size_model = get_size(copy.deepcopy(model))
            optimizer = optim.Adam(model.parameters(), lr=lr_training)
            model, loss, acc, st, time_elapsed2 = trainer.trainer(
                model, optimizer, trainer.criterion, num_epochs=N_it_ob, ind=f"it{it}"
            )
            load = (
                snp
                + "/"
                + modelName
                + "_it_{}_acc_{:.3f}_size_{:.3f}.pth".format(
                    it, acc, size_model / start_size_model
                )
            )
            # Кастыль для сегментации в офе
            if (
                config["model"]["type_save_load"] == "interface"
                and config["task"]["type"] == "segmentation"
            ):
                model.backbone_hooks._clear_hooks()
            torch.save(model, load)
            f = open(exp_save + "/" + modelName + "_log.txt", "a")
            f.write(
                "N {} sloi {} do {} posle {} acc {} size {}\n".format(
                    it, "pass", "[ ]", "[ ]", acc, size_model / start_size_model
                )
            )
            f.close()
            for filename in os.listdir(snp):
                if filename.split(".")[-1] == "pth":
                    if (
                        len(filename.split("size")) == 1
                        and filename != "orig_model.pth"
                    ):
                        os.remove(snp + "/" + filename)
                if filename.split(".")[-1] == "txt":
                    if len(filename.split("train_log")) == 2:
                        os.remove(snp + "/" + filename)

    it = start_iteration
    stract = get_stract(model)
    size_model = get_size(model)
    del model
    while start_size_model * (1 - P) < size_model:
        print(start_size_model, size_model, start_size_model * (1 - P))
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
                    p = Process(target=potok, args=parametri[opit])
                    p.start()
                    rab.append(p)
                    opit += 1
            for p in rab:
                p.join()
            time_elapsed = time.time() - since2
            print(
                "time_iter= {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

        fil_it = snp + "/" + modelName + "_it{}.txt".format(it)
        f = open(fil_it, "r")
        strr = f.read()
        strr = strr.split("\n")
        old_load = load
        for st in strr[:-1]:
            if st.split(" ")[2] != "EROR":
                maxx = float(st.split(" ")[2])
                load = (
                    snp
                    + "/"
                    + modelName
                    + "_"
                    + st.split(" ")[1]
                    + "_it_{}_acc_{:.3f}.pth".format(it, maxx)
                )
                sloi = st.split(" ")[1]
                do = st.split(" ")[3] + " " + st.split(" ")[4]
                posle = st.split(" ")[5] + " " + st.split(" ")[6]
                break
        for st in strr[:-1]:
            if st.split(" ")[2] != "EROR":
                if float(st.split(" ")[2]) > maxx:
                    maxx = float(st.split(" ")[2])
                    load = (
                        snp
                        + "/"
                        + modelName
                        + "_"
                        + st.split(" ")[1]
                        + "_it_{}_acc_{:.3f}.pth".format(it, maxx)
                    )
                    sloi = st.split(" ")[1]
                    do = st.split(" ")[3] + " " + st.split(" ")[4]
                    posle = st.split(" ")[5] + " " + st.split(" ")[6]
        if old_load == load:
            break
        model = torch.load(load)
        m = copy.deepcopy(model)
        # Кастыль для сегментации в офе
        if (
            config["model"]["type_save_load"] == "interface"
            and config["task"]["type"] == "segmentation"
        ):
            model.backbone_hooks._attach_hooks()
            m.backbone_hooks._attach_hooks()

        optimizer = optim.Adam(model.parameters(), lr=lr_training)
        size_model = get_size(m)
        del m
        model, loss, acc, st, time_elapsed2 = trainer.trainer(
            model, optimizer, trainer.criterion, num_epochs=N_it_ob, ind=f"it{it}"
        )
        load = (
            snp
            + "/"
            + modelName
            + "_it_{}_acc_{:.3f}_size_{:.3f}.pth".format(
                it, acc, size_model / start_size_model
            )
        )
        # Кастыль для сегментации в офе
        if (
            config["model"]["type_save_load"] == "interface"
            and config["task"]["type"] == "segmentation"
        ):
            model.backbone_hooks._clear_hooks()
        torch.save(model, load)
        stract = get_stract(model)
        del model
        f = open(exp_save + "/" + modelName + "_log.txt", "a")
        f.write(
            "N {} sloi {} do {} posle {} acc {} size {}\n".format(
                it, sloi, do, posle, acc, size_model / start_size_model
            )
        )
        f.close()
        for filename in os.listdir(snp):
            if filename.split(".")[-1] == "pth":
                if len(filename.split("size")) == 1 and filename != "orig_model.pth":
                    os.remove(snp + "/" + filename)
            if filename.split(".")[-1] == "txt":
                if len(filename.split("train_log")) == 2:
                    os.remove(snp + "/" + filename)
        it = it + 1
        time_elapsed = time.time() - since3
        print(
            "time_epox= {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
    time_elapsed = time.time() - since
    print("time_total= {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print(start_size_model, size_model, start_size_model * (1 - P))
