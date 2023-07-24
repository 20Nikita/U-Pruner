from pydantic import BaseModel
from typing import Literal
TaskTypes = Literal[
    "segmentation",
    "detection",
    "classification",
    "special points"
]
LOAD_VARIANTS = Literal[
    "pth",
    "interface"
]

LOAD_ALGRORITHMS = Literal[
    ALG1
    ALG2,
]

class Task(BaseModel):
    type: TaskTypes
    annotation_path:str
    annotation_name_train:str
    annotation_name_val:str
    
class Dataset(BaseModel):
    num_classes:int
    

class PathConfig(BaseModel):
    exp_save:str = "/workspace/snp"  
    model_name:str = "Detect_2"
    
class ModelConfig(BaseModel):
    type_save_load: LOAD_VARIANTS

class Config(BaseModel):
    model: ModelConfig
    path: PathConfig

    
DEFAULT_CONFIG_PATH = "Pruning.yaml"

