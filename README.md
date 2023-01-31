# Pruning

### Description
Инструмент обрезки моделей.

Реализованы алгоритмы: [my_pruning](#my_pruning), [L2Norm](#L2Norm)

### my_pruning
my_pruning - алгоритм обрезки, основанный на [NetAdapt](https://arxiv.org/abs/1804.03230).

Алгоритм разбивает сеть на блоки c помочью model.named_parameters(), обрезает каждый блок низкоуровневым алгоритмом TaylorFOWeight или L2Norm, до обучает и выбирает лучший по Acc. Так по 1 блоку постепенно обрезает всю сеть. Поддерживается обрезка torch.nn.modules.conv.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.linear.Linear.

#### Гиппер параметры
- alf        - Сохранить кратность первых двух каналов конволюции числу alf (33*65*7*7)->(32*64*7*7)
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- cart       - На каких картах обучать [0,0,1,1,1,2] (для параллельной работы, запустится обрезка 6 блоков с тренировкой на соответствующих картах)
- iskl       - Название слоёв, которые не нужно обрезать [input_stem.0.conv,blocks.1.conv2.conv]
- algoritm   - TaylorFOWeight, L2Norm. Прунинг в основе. TaylorFOWeight в 2 раза дольче L2Norm но немного лучше.
- resize_alf - Обрезать всю сеть до кратности alf (Если False то могут остаться не кратные свертки при условии что их обрезка сильно портит качество)
- delta_crop - Сколько % от блока резать за 1 итерацию 0.1->(100*100*7*7)->(90->100*7*7)

### L2Norm
L2Norm - мгновенный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#l1-norm-pruner) и доработан, для поддержки авто генерируемых сетей.
#### Гиппер параметры
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training:  - До обучить после обрезки

# Example
    workspace
    ├── README.md
    ├── interfaces
    │   └── *
    ├── dtools   
    │   ├── Dockerfile
    │   ├── .dockerignore
    │   ├── docker_run.sh
    │   └── build.sh 
    ├── "exp_save" # save loog config["path"]["exp_save"] (/storage/3030/GordeevN/prj/Pruning/snp/Mypruning)
    │   ├── "model_name" config["path"]["model_name"]
    │   │    └── * # loog
    │   └── "model_name".txt
    └── proj
        ├── Pruning.yaml
        ├── main.py
        ├── requirements.txt
        ├── cod
        │   ├── dataset.py
        │   ├── ModelSpeedup.py
        │   ├── my_pruning_pabotnik.py
        │   ├── my_pruning.py
        │   ├── retraining.py
        │   ├── train_nni.py
        │   └── training.py
        ├── shared # interface
        │   └── results
        │       └── interface.pt
        └── dataset # dataset  (/storage_labs/db/paradigma/chest_xray/dataset_chest_xray_640_480)
            ├── *
            └── data.csv
            
