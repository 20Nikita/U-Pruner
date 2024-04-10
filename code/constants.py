from pydantic import BaseModel
from typing import Literal, List, Union, Any

LOAD_VARIANTS = Literal["pth", "interface"]

TaskTypes = Literal["segmentation", "detection", "classification", "special points"]
DetectionTypes = Literal["ssd", "yolo", None]

LossTypes = Literal["CustomBCELoss", None]
MetricsTypes = Literal["SMPMetric", None]


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
ClassName = Union[None, List[str]]


class PathConfig(BaseModel):
    exp_save: str = "/workspace/snp"
    modelName: str = "Detect_2"


class ModelConfig(BaseModel):
    type_save_load: LOAD_VARIANTS = "pth"
    path_to_resurs: str = "/workspace/proj/shared/results"
    name_resurs: str = "interface"
    size: List[int] = [224, 224]
    gpu: int = 0
    anchors: Union[None, List[List[List[int]]]] = None
    feature_maps_w: Union[None, List[int]] = None
    feature_maps_h: Union[None, List[int]] = None
    aspect_ratios: Union[None, List[List[int]]] = None


class Task(BaseModel):
    type: TaskTypes = "classification"
    detection: DetectionTypes = None
    loss: LossTypes = None
    metrics: MetricsTypes = None


class Mask(BaseModel):
    type: Literal["mask", None] = "mask"
    sours_mask: Union[None, str] = None


class Dataset(BaseModel):
    num_classes: int = 10
    annotation_path: str = ""
    annotation_name: Union[None, str] = None
    annotation_name_train: Union[None, str] = None
    annotation_name_val: Union[None, str] = None


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
    is_self_traner: bool = False
    self_traner: str = ''
    num_epochs: int = 1
    lr: float = 0.00001
    dataLoader: DataLoader = DataLoader()


class Training(BaseModel):
    is_self_traner: bool = False
    self_traner: str = ''
    num_epochs: int = 1
    lr: float = 0.00001
    dataLoader: DataLoader = DataLoader()


class Restart(BaseModel):
    start_iteration: int = 0
    load: Union[None, str] = None


class MyPruning(BaseModel):
    alf: int = 32
    P: float = 0.8
    cart: List[int] = [0]
    top_n: int = 1
    iskl: List[str] = []
    algoritm: MyPruningAlgorithmTypes = "TaylorFOWeight"
    resize_alf: bool = False
    delta_crop: float = 0.1
    restart: Restart = Restart()


class NniPruning(BaseModel):
    P: float = 0.5
    training: bool = True
    total_iteration: int = 10
    gpu: int = 0


class Config(BaseModel):
    path: PathConfig = PathConfig()
    class_name: ClassName = None
    model: ModelConfig = ModelConfig()
    task: Task = Task()
    mask: Mask = Mask()
    dataset: Dataset = Dataset()
    retraining: Retraining = Retraining()
    training: Training = Training()
    algorithm: AlgorithmTypes = "My_pruning"
    my_pruning: MyPruning = MyPruning()
    nni_pruning: NniPruning = NniPruning()


# DEFAULT_CONFIG_PATH = "configs/derection_SSD.yaml"
# DEFAULT_CONFIG_PATH = "configs/classification_timm.yaml"
# DEFAULT_CONFIG_PATH = "configs/detection_Yolo.yaml"
# DEFAULT_CONFIG_PATH = "configs/classification-paradigma.yaml"
# DEFAULT_CONFIG_PATH = "configs/Wclassification_timm.yaml"
# DEFAULT_CONFIG_PATH = "configs/Wclassification.yaml"
# DEFAULT_CONFIG_PATH = "configs/classification_TR50_Caltech256.yaml"
DEFAULT_CONFIG_PATH = "config.yaml"