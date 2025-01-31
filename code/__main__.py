import os
import yaml
import pandas as pd
from constants import DEFAULT_CONFIG_PATH, Config
from argparse import ArgumentParser

log = open("log.txt", "w")
log.close()
parser = ArgumentParser()
parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH)

args = parser.parse_args()
print(args.config)

config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
config = Config(**config)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.model.gpu)


def main(config: Config):
    import torch
    import copy
    import time
    import torch.optim as optim
    import importlib

    import nni

    import my_pruning as my_pruning
    from my_pruning_pabotnik import get_size, get_stract
    from ModelSpeedup import ModelSpeedup
    from utils_torch import ALGORITHMS

    if config.training.is_self_traner:
        trainer = importlib.import_module(config.training.self_traner)
    else:
        trainer = importlib.import_module('training')

    since = time.time()
    acc = 0
    component = "pruning"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    snp = os.path.join(config.path.exp_save, config.path.modelName)
    if not os.path.exists(snp):
        os.makedirs(snp)
    log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
    log.write(str(args.config) + "\n")
    log.close()
    if config.model.type_save_load == "interface":
        from interfaces.tools import load_interface, build_net, save_interface
        # trainer = importlib.import_module('training')

        interface = load_interface(
            path_to_interface=config.model.path_to_resurs,
            load_name=config.model.name_resurs,
        )
        model = build_net(interface=interface, pretrained=True)
        # model = build_net(interface=interface, pretrained = False)
    elif config.model.type_save_load == "pth":
        import sys

        sys.path.append(config.model.path_to_resurs)
        model = torch.load(
            os.path.join(
                config.model.path_to_resurs, f"{config.model.name_resurs}.pth"
            ),
            map_location=device,
        )
        # if config.training.is_self_traner:
        #     trainer = importlib.import_module(config.training.self_traner)
        # else:
        #     trainer = importlib.import_module('training')

    print(config.algorithm)
    log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
    log.write(str(config.algorithm) + "\n")
    log.close()

    if config.algorithm == "My_pruning":
        # Кастыль для хуков в офе
        if config.model.type_save_load == "interface":
            model.backbone_hooks._clear_hooks()
        torch.save(model, os.path.join(snp, "orig_model.pth"))
        # Кастыль для хуков в офе
        if config.model.type_save_load == "interface":
            model.backbone_hooks._attach_hooks()
        start_size_model = get_size(model, config.model.size)
        del model
        my_pruning.my_pruning(
            start_size_model=start_size_model, config_path=args.config
        )
        data = pd.read_csv(os.path.join(config.path.exp_save, config.path.modelName + "_log.csv"))
        N, sloi, do, posle, acc, size = data.iloc[-1].values
        model = torch.load(
            os.path.join(
                config.path.exp_save,
                config.path.modelName,
                f"{config.path.modelName}_it_{N}_acc_{float(acc):.3f}_size_{float(size):.3f}.pth",
            )
        )
        

    else:
        model = model.to(device)
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        config_list = [{"sparsity": config["nni_pruning"]["P"], "op_types": ["Conv2d"]}]
        pruning_params = {
            "trainer": trainer.retrainer,
            "traced_optimizer": traced_optimizer,
            "criterion": trainer.criterion,
            "training_batches": config["dataset"]["dataLoader"]["batch_size_t"],
        }
        pruner = ALGORITHMS[config["algorithm"]](model, config_list, pruning_params)

        if (
            config.algorithm == "L2Norm"
            or config.algorithm == "FPGM"
            or config.algorithm == "TaylorFOWeight"
        ):
            masked_model, masks = pruner.compress()
        else:
            pruner.compress()
            _, model, masks, _, _ = pruner.get_best_result()
        pruner._unwrap_model()
        model = ModelSpeedup(model, masks).to(device)
        if config.nni_pruning.training:
            optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
            model, loss, acc, st, time_elapsed2 = trainer.trainer(
                model,
                optimizer,
                trainer.criterion,
                num_epochs=config.training.num_epochs,
            )

    if config.model.type_save_load == "interface":
        model_orig = build_net(interface=interface)
        model_prun = model
        stract1 = get_stract(model_orig)
        stract2 = get_stract(model_prun)

        m = copy.deepcopy(model_prun)
        mo = copy.deepcopy(model_orig)
        if config.model.type_save_load == "interface":
            m.backbone_hooks._attach_hooks()
            mo.backbone_hooks._attach_hooks()
        size = get_size(m, config.model.size) / get_size(mo, config.model.size)
        params = model_prun.state_dict()
        if ("pruning" in interface) and len(interface.pruning.summary.resize):
            resize = interface.pruning.summary.resize
        else:
            resize = []
            interface["pruning"] = {}
            interface["pruning"]["summary"] = {}
            interface["pruning"]["is_pruning"] = True
        for i in range(len(stract1)):
            if stract1[i][2] != stract2[i][2]:
                resize.append(
                    {
                        "name": stract1[i][0],
                        "type": stract1[i][1],
                        "orig_shape": stract1[i][2],
                        "shape": stract2[i][2],
                    }
                )

        time_elapsed = time.time() - since
        summary = {
            "size": float(size),
            "val_accuracy": float(acc),
            "resize": resize,
            "time": f"{time_elapsed // 60 // 60:.0f}h {time_elapsed // 60 - time_elapsed // 60 // 60 * 60:.0f}m {time_elapsed % 60:.0f}s",
        }

        save_interface(
            params=params,
            interface=interface,
            name_component=component,
            summary=summary,
            path_to_interface=config.model.path_to_resurs,
        )
    elif config.model.type_save_load == "pth":
        torch.save(model, os.path.join(config.model.path_to_resurs, "rezalt.pth"))

    print('End!')
    log = open(os.path.join(config.path.exp_save, "log.txt"), "a")
    log.write("End!\n")
    log.close()


if __name__ == "__main__":
    main(config)
