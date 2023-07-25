from pydantic import BaseModel
from typing import Literal
LOAD_VARIANTS = Literal[
    "pth",
    "interface"
]

TaskTypes = Literal[
    "segmentation",
    "detection",
    "classification",
    "special points"
]
MaskTypes = Literal[
    "mask",
    "None"
]
SourseMaskTypes = Literal[
    list(list),
    "None"
]
AnnotationNameTypes = Literal[
    None,
    str
]
AlgorithmTypes = Literal[
    "My_pruning", 
    "AGP", 
    "Linear", 
    "LotteryTicket", 
    "TaylorFOWeight", 
    "ActivationMeanRank", 
    "FPGM", 
    "L2Norm"
]
MyPruningAlgorithmTypes = Literal[
    "TaylorFOWeight", 
    "L2Norm"
]

class PathConfig(BaseModel):
    exp_save:str = "/workspace/snp"  
    model_name:str = "Detect_2"
    
class ModelConfig(BaseModel):
    type_save_load: LOAD_VARIANTS
    path_to_resurs:str = "/workspace/proj/shared/results"
    name_resurs:str = "interface"
    size:list(int,int) = [224,224]
    gpu: int = 0
    anchors: list(list(int,int))
    

class Task(BaseModel):
    type: TaskTypes
    
class Mask(BaseModel):
    type: MaskTypes
    sours_mask: SourseMaskTypes

class Dataset(BaseModel):
    num_classes:int
    annotation_path:str
    annotation_name:AnnotationNameTypes
    annotation_name_train:str
    annotation_name_val:str

class DataLoader(BaseModel):
    batch_size_t:int = 10
    num_workers_t:int =  0
    pin_memory_t:bool = True
    drop_last_t:bool = True
    shuffle_t:bool = True

    batch_size_v:int = 10
    num_workers_v:int = 0
    pin_memory_v:bool = True
    drop_last_v:bool = True
    shuffle_v:bool = False

class Retraining(BaseModel):
    num_epochs: int = 1
    lr: float = 0.00001
    dataLoader:DataLoader

class Training(BaseModel):
    num_epochs: int = 1
    lr: float = 0.00001
    dataLoader:DataLoader    

class algorithm(BaseModel):
    algorithm: AlgorithmTypes 
    
class Restart(BaseModel):
    start_iteration:int = 0
    load:str = PathConfig.exp_save + '/' + PathConfig.model_name + '/orig_model.pth'

class MyPruning(BaseModel):
    alf: int = 32 
    P:float = 0.8
    cart: list(int)
    iskl: list(str)
    algoritm: MyPruningAlgorithmTypes
    resize_alf:bool = False
    delta_crop:float = 0.1
    restart:Restart

class NniPruning(BaseModel):
    P:float = 0.5 
    training:bool = True
    total_iteration:int = 10
    gpu:int = 0

class Config(BaseModel):
    path: PathConfig
    model: ModelConfig
    task: Task
    mask: Mask
    dataset:Dataset
    retraining:Retraining
    training:Training
    algorithm: AlgorithmTypes
    my_pruning:MyPruning
    nni_pruning:NniPruning

    
DEFAULT_CONFIG_PATH = "Pruning.yaml"

