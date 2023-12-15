import torch
import cv2
import os
import json
import yaml

import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import Dataset
from torchvision.ops import box_convert, nms

import albumentations as A
import albumentations.pytorch as Ap
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from constants import DEFAULT_CONFIG_PATH, Config
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH)
args, unknown = parser.parse_known_args()
config = yaml.safe_load(open(args.config, encoding="utf-8"))
config = Config(**config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_class = config.dataset.num_classes
head_anchors = config.model.anchors
D_SP = []
if config.task.type == "detection" and config.task.detection == "ssd":
    for i in config.model.feature_maps_w:
        D_SP.append(config.model.size[0] / i)


def get_ransforms(width, height):
    train_transform = A.Compose(
        [
            A.Resize(width=int(width * 1.2), height=int(height * 1.2)),
            A.RandomCrop(width=width, height=height),
            A.Flip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            Ap.transforms.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco"),
    )

    val_transform = A.Compose(
        [
            A.Resize(width=int(width * 1.2), height=int(height * 1.2)),
            A.CenterCrop(width=width, height=height),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            Ap.transforms.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco"),
    )
    return train_transform, val_transform


class MyDataset(Dataset):
    def __init__(self, annotation, root, boof, transform=None):
        data = json.load(open(annotation))
        self.transform = transform
        self.root = root
        self.boof = boof
        self.images = pd.DataFrame(data["images"])
        self.annotations = pd.DataFrame(data["annotations"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(
            self.root, self.boof, self.images.iloc[idx][["file_name"]].item()
        )
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        idd = self.images.iloc[idx][["id"]].item()
        data = self.annotations[self.annotations.image_id == idd][
            ["bbox", "category_id"]
        ].values
        labels = []
        for i, label in enumerate(data):
            labels.append(label[0])
            labels[-1].append(label[1])
            # labels[-1].append(idd )
        # print(labels)
        if self.transform:
            transformed = self.transform(image=image, bboxes=labels)
            image = transformed["image"]
            labels = transformed["bboxes"]
        return image, labels


def display_images(l, titles=None, fontsize=12, labels=None, tru_labels=None):
    n = len(l)
    fig, ax = plt.subplots(1, n)
    for i, im in enumerate(l):
        im = im.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = std * im + mean
        ax[i].imshow(im)
        ax[i].axis("off")
        t = []
        if labels is not None:
            for label in labels[i]:
                if len(label):
                    ax[i].add_patch(
                        patches.Rectangle(
                            (int(label[0]), int(label[1])),
                            int(label[2]),
                            int(label[3]),
                            edgecolor="red",
                            facecolor="none",
                            lw=4,
                        )
                    )
                    t.append(label[4])
                    ax[i].annotate(
                        label[4], (int(label[0]), int(label[1])), fontsize=16
                    )
        if tru_labels is not None:
            for label in tru_labels[i]:
                if len(label):
                    ax[i].add_patch(
                        patches.Rectangle(
                            (int(label[0]), int(label[1])),
                            int(label[2]),
                            int(label[3]),
                            edgecolor="green",
                            facecolor="none",
                            lw=4,
                        )
                    )
                    t.append(label[4])
                    ax[i].annotate(
                        label[4], (int(label[0]), int(label[1])), fontsize=16
                    )
    fig.set_size_inches(fig.get_size_inches() * n)
    plt.tight_layout()
    plt.show()


def custom_collate(batch):
    images, labels = [], []
    for b in batch:
        images.append(b[0])
        labels.append(b[1])
    images = torch.stack(images)
    return images, labels


def get_component(label):
    m = (
        min(label[2], head_anchors[0][0][0]) * min(label[3], head_anchors[0][0][1])
    ) / (max(label[2], head_anchors[0][0][0]) * max(label[3], head_anchors[0][0][1]))
    mi = 0
    mj = 0
    for i, v in enumerate(head_anchors):
        for j, d in enumerate(v):
            km = (min(label[2], d[0]) * min(label[3], d[1])) / (
                max(label[2], d[0]) * max(label[3], d[1])
            )
            if m < km:
                m = km
                mi = i
                mj = j
        if config.task.detection == "ssd":
            SP = D_SP[mi]
        else:
            SP = 2 ** (mi + 1)
        a = int(label[0] // SP)
        b = int(label[1] // SP)
    return [mi, mj, a, b, SP]


def get_pred(out_model, k=10, alf=0.2, iou_threshold=0):
    pred = dict(
        boxes=torch.tensor([[]], dtype=torch.float, device=device),
        scores=torch.tensor([], dtype=torch.float, device=device),
        labels=torch.tensor([], dtype=torch.float, device=device),
    )

    for mi, d1 in enumerate(out_model):
        d1 = torch.transpose(d1, 2, 0)
        sh = d1.shape[0]
        t = torch.reshape(d1, (-1, N_class + 5))
        index = torch.topk(t[:, 0].flatten(), k).indices
        for i in index:
            if torch.sigmoid(t[i][0]) > alf:
                mj = i.item() % len(head_anchors[mi])
                a = i.item() // len(head_anchors[mi]) % sh
                b = i.item() // len(head_anchors[mi]) // sh
                if config.task.detection == "ssd":
                    SP = D_SP[mi]
                else:
                    SP = 2 ** (mi + 1)
                pred["scores"] = torch.cat(
                    (pred["scores"], t[i][0].unsqueeze(0)), dim=0
                )
                pred["labels"] = torch.cat(
                    (pred["labels"], torch.argmax(t[i][5:]).unsqueeze(0) + 1), dim=0
                )
                pred["boxes"] = torch.cat(
                    (
                        pred["boxes"],
                        torch.cat(
                            (
                                (torch.sigmoid(t[i][1].unsqueeze(0)) + a) * SP,
                                (torch.sigmoid(t[i][2].unsqueeze(0)) + b) * SP,
                                torch.exp(t[i][3].unsqueeze(0))
                                * head_anchors[mi][mj][0],
                                torch.exp(t[i][4].unsqueeze(0))
                                * head_anchors[mi][mj][1],
                            ),
                            dim=0,
                        ).unsqueeze(0),
                    ),
                    dim=int(pred["boxes"].size()[1] == 0),
                )
    if pred["boxes"].size()[1] == 0:
        pred["boxes"] = torch.tensor([], dtype=torch.float, device=device)
    else:
        ind = nms(
            boxes=box_convert(boxes=pred["boxes"], in_fmt="xywh", out_fmt="xyxy"),
            scores=pred["scores"],
            iou_threshold=iou_threshold,
        )
        pred["boxes"] = pred["boxes"][ind]
        pred["labels"] = pred["labels"][ind]
        pred["scores"] = pred["scores"][ind]
    return pred


def get_preds(outs_model, k=10, alf=0.2, iou_threshold=0):
    rezalt = []
    for i in range(outs_model[0].shape[0]):
        out = []
        for d in outs_model:
            out.append(d[i])
        rezalt.append(get_pred(out, k, alf, iou_threshold))
    return rezalt


def get_target(labels):
    rezalt = []
    for label in labels:
        target = dict(
            boxes=torch.tensor([[]], dtype=torch.float, device=device),
            labels=torch.tensor([], dtype=torch.float, device=device),
        )
        for l in label:
            target["labels"] = torch.cat(
                (
                    target["labels"],
                    torch.tensor(l[4], dtype=torch.float, device=device).unsqueeze(0),
                ),
                dim=0,
            )
            target["boxes"] = torch.cat(
                (
                    target["boxes"],
                    torch.tensor([l[:4]], dtype=torch.float, device=device),
                ),
                dim=int(target["boxes"].size()[1] == 0),
            )
        if target["boxes"].size()[1] == 0:
            target["boxes"] = torch.tensor([], dtype=torch.float, device=device)
        rezalt.append(target)
    return rezalt


BCELoss = nn.BCELoss()
BCEobj = nn.BCEWithLogitsLoss()
CrossEntropyLoss = nn.CrossEntropyLoss()
MSELoss = nn.MSELoss()
Softmax = nn.Softmax(dim=0)
cof_bloss = 5000


def Loss(out, labels):
    ver = []
    for o in out:
        ver.append(torch.zeros(o.shape))
    bloss = []
    bceloss = []
    xyloss = []
    vhloss = []
    clasloss = []
    for bath, l1 in enumerate(labels):
        for l in l1:
            mi, mj, a, b, SP = get_component(l)
            ind = mj * (1 + N_class + 4)
            indk = (mj + 1) * (1 + N_class + 4)
            xyloss.append(
                BCELoss(
                    torch.sigmoid(out[mi][bath][ind + 1 : ind + 3, a, b]),
                    torch.tensor(
                        [l[0] % SP / SP, l[1] % SP / SP],
                        dtype=torch.float,
                        device=device,
                    ),
                )
            )
            xyloss.append(
                MSELoss(
                    out[mi][bath][ind + 3, a, b],
                    torch.log(
                        torch.tensor(
                            l[2] / head_anchors[mi][mj][0],
                            dtype=torch.float,
                            device=device,
                        )
                    ),
                )
            )
            xyloss.append(
                MSELoss(
                    out[mi][bath][ind + 4, a, b],
                    torch.log(
                        torch.tensor(
                            l[3] / head_anchors[mi][mj][1],
                            dtype=torch.float,
                            device=device,
                        )
                    ),
                )
            )
            clasloss.append(
                CrossEntropyLoss(
                    out[mi][bath][ind + 5 : indk, a, b],
                    torch.tensor(l[4] - 1, device=device),
                )
            )
            bceloss.append(
                BCELoss(
                    torch.sigmoid(out[mi][bath, ind, a, b]),
                    torch.tensor(1, dtype=torch.float, device=device),
                )
            )

    for l, o in zip(ver, out):
        bloss.append(
            BCEobj(o[:, :: N_class + 5, :, :], l[:, :: N_class + 5, :, :].to(device))
        )
    bceloss = sum(bceloss) / len(bceloss)
    bloss = sum(bloss) / len(bloss)
    xyloss = sum(xyloss) / len(xyloss)
    clasloss = sum(clasloss) / len(clasloss)
    return bloss * cof_bloss + xyloss + clasloss + bceloss


class Decod:
    def __init__(self):
        self.num_classes = config.dataset.num_classes + 1
        self.num_layers = len(config.model.aspect_ratios)
        self.init_sizes = [0] * (self.num_layers * 2)
        for i in range(len(self.init_sizes)):
            if i < len(self.init_sizes) / 2:  # loc feature map sizes
                num_anchors_there = len(config.model.aspect_ratios[i]) * 2 + 2
                f_h = config.model.feature_maps_h[i]
                f_w = config.model.feature_maps_w[i]
                self.init_sizes[i] = num_anchors_there * 4 * f_h * f_w
            else:  # conf feature map sizes
                num_anchors_there = (
                    len(config.model.aspect_ratios[i % self.num_layers]) * 2 + 2
                )
                f_h = config.model.feature_maps_h[i % self.num_layers]
                f_w = config.model.feature_maps_w[i % self.num_layers]
                self.init_sizes[i] = num_anchors_there * self.num_classes * f_h * f_w

    def decod(self, predictions):
        preds_split = torch.split(predictions, self.init_sizes, dim=1)
        dims = [0] * (self.num_layers * 2)
        batch_size = predictions.shape[0]
        for i in range(len(dims)):
            if i < len(dims) / 2:  # loc feature map shapes
                num_anchors_there = len(config.model.aspect_ratios[i]) * 2 + 2
                f_h = config.model.feature_maps_h[i]
                f_w = config.model.feature_maps_w[i]
                dims[i] = [batch_size, num_anchors_there * 4, f_h, f_w]
            else:  # conf feature map shapes
                num_anchors_there = (
                    len(config.model.aspect_ratios[i % self.num_layers]) * 2 + 2
                )
                f_h = config.model.feature_maps_h[i % self.num_layers]
                f_w = config.model.feature_maps_w[i % self.num_layers]
                dims[i] = [batch_size, num_anchors_there * self.num_classes, f_h, f_w]
        preds = []
        for i, pred in enumerate(preds_split):
            preds.append(pred.view(dims[i]))

        loc_data = preds[: self.num_layers]
        conf_data = preds[self.num_layers :]
        out = []

        for i, j in zip(loc_data, conf_data):
            out.append(torch.cat((i, j), dim=1))
        return out
