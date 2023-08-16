from pydantic import BaseModel
from typing import Literal, List, Union

LOAD_VARIANTS = Literal["pth", "interface"]

TaskTypes = Literal["segmentation", "detection", "classification", "special points"]
DetectionTypes = Literal[
    "ssd",
    "yolo",
]
MaskTypes = Literal["mask", "None"]
SourseMaskTypes = Literal[List[List[int]], "None"]
AlgorithmTypes = Literal[
    "My_pruning",
    "AGP",
    "Linear",
    "LotteryTicket",
    "TaylorFOWeight",
    "ActivationMeanRank",
    "FPGM",
    "L2Norm",
]
MyPruningAlgorithmTypes = Literal["TaylorFOWeight", "L2Norm"]


class PathConfig(BaseModel):
    exp_save: str = "/workspace/snp"
    modelName: str = "Detect_2"


class ModelConfig(BaseModel):
    type_save_load: LOAD_VARIANTS
    path_to_resurs: str = "/workspace/proj/shared/results"
    name_resurs: str = "interface"
    size: List[int] = [224, 224]
    gpu: int = 0
    anchors: Union[Literal[None], List[List[List[int]]]]
    feature_maps_w: Union[Literal[None], List[int]]
    feature_maps_h: Union[Literal[None], List[int]]
    aspect_ratios: Union[Literal[None], List[List[int]]]


class Task(BaseModel):
    type: TaskTypes
    detection: Literal[DetectionTypes, None]


class Mask(BaseModel):
    type: MaskTypes
    sours_mask: SourseMaskTypes


class Dataset(BaseModel):
    num_classes: int
    annotation_path: str
    annotation_name: Union[Literal[None], str]
    annotation_name_train: Union[Literal[None], str]
    annotation_name_val: Union[Literal[None], str]


class DataLoader(BaseModel):
    batch_size_t: int = 10
    num_workers_t: int = 0
    pin_memory_t: bool = True
    drop_last_t: bool = True
    shuffle_t: bool = True

    batch_size_v: int = 10
    num_workers_v: int = 0
    pin_memory_v: bool = True
    drop_last_v: bool = True
    shuffle_v: bool = False


class Retraining(BaseModel):
    num_epochs: int = 1
    lr: float = 0.00001
    dataLoader: DataLoader


class Training(BaseModel):
    num_epochs: int = 1
    lr: float = 0.00001
    dataLoader: DataLoader


class algorithm(BaseModel):
    algorithm: AlgorithmTypes


class Restart(BaseModel):
    start_iteration: int = 0
    load: str  # = PathConfig.exp_save + '/' + PathConfig.modelName + '/orig_model.pth'


class MyPruning(BaseModel):
    alf: int = 32
    P: float = 0.8
    cart: List[int]
    iskl: List[str]
    algoritm: MyPruningAlgorithmTypes
    resize_alf: bool = False
    delta_crop: float = 0.1
    restart: Restart


class NniPruning(BaseModel):
    P: float = 0.5
    training: bool = True
    total_iteration: int = 10
    gpu: int = 0


class Config(BaseModel):
    path: PathConfig
    model: ModelConfig
    task: Task
    mask: Mask
    dataset: Dataset
    retraining: Retraining
    training: Training
    algorithm: AlgorithmTypes
    my_pruning: MyPruning
    nni_pruning: NniPruning


# DEFAULT_CONFIG_PATH = "configs/derection_SSD.yaml"
DEFAULT_CONFIG_PATH = "configs/classification_timm.yaml"
# DEFAULT_CONFIG_PATH = "configs/detection_Yolo.yaml"
# DEFAULT_CONFIG_PATH = "configs/classification-paradigma.yaml"
