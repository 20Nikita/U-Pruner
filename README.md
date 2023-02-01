# Pruning

## Description
Инструмент обрезки моделей.

Реализованы алгоритмы: [my pruning](#my-pruning), [L2Norm](#l2norm), [FPGM](#fpgm), [TaylorFOWeight](#taylorfoweight), [FPGM](#fpgm), [AGP](#agp), [Linear](#linear), [LotteryTicket](#lotteryticket).
## my pruning
my_pruning - алгоритм обрезки, основанный на [NetAdapt](https://arxiv.org/abs/1804.03230). Разработан специально для генерируемых сетей алгоритмами NAS. Занимает много времени, но работает одинаково хорошо с любой архитектурой сети и любым степенем сжатия.

Алгоритм разбивает сеть на блоки c помочью model.named_parameters(), обрезает каждый блок низкоуровневым алгоритмом TaylorFOWeight или L2Norm, до обучает и выбирает лучший по Acc. Так по 1 блоку постепенно обрезает всю сеть. 

Поддерживается обрезка torch.nn.modules.conv.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.linear.Linear.

#### Гиппер параметры
- alf        - Сохранить кратность первых двух каналов конволюции числу alf (33*65*7*7)->(32*64*7*7)
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- cart       - На каких картах обучать [0,0,1,1,1,2] (для параллельной работы, запустится обрезка 6 блоков с тренировкой на соответствующих картах)
- iskl       - Название слоёв, которые не нужно обрезать [input_stem.0.conv,blocks.1.conv2.conv]
- algoritm   - TaylorFOWeight, L2Norm. Прунинг в основе. TaylorFOWeight в 2 раза дольче L2Norm но немного лучше.
- resize_alf - Обрезать всю сеть до кратности alf (Если False то могут остаться не кратные свертки при условии что их обрезка сильно портит качество)
- delta_crop - Сколько % от блока резать за 1 итерацию 0.1->(100*100*7*7)->(90->100*7*7)

## Мгновенные алгоритмы обрезки
[L2Norm](#l2norm), [FPGM](#fpgm) - мгновенные алгоритмы обрезки. Идеальный ваниант если нужно отрезать до 10% от сети.

#### Гиппер параметры
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training:  - До обучить после обрезки

### L2Norm
L2Norm - мгновенный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#l1-norm-pruner) и доработан, для поддержки авто генерируемых сетей.

Вычисляет L2 норму в 1 измерении свёрточных слоёв и удаляет строки с наименьшим значением этой метрики.

Особенно хорошо справляется с архитектурами подобными mobilenetv2

### FPGM
FPGM - мгновенный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#fpgm-pruner) и доработан, для поддержки авто генерируемых сетей.

Вычисляет медиану в 1 измерении свёрточных слоёв и удаляет строки с наименьшим значением этой метрики.

Особенно хорошо справляется с архитектурами подобными resnet

## Одностадийные алгоритмы обрезки
Одностадийные алгоритмы обрезки нуждаются в информации об инференсе модели. Требуется пройти 1 эпоху обучения для сбора данных перед обрезкой, но благодоря этому алгоритмы имеют гораздо больший потенциал.

#### Гиппер параметры
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training:  - До обучить после обрезки

### TaylorFOWeight
TaylorFOWeight - одностадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#nni.compression.pytorch.pruning.TaylorFOWeightPruner) и доработан, для поддержки авто генерируемых сетей.

Работает на основе оценочной важности, рассчитанной из разложения Тейлора первого порядка для 1 измерении весов свёрточных слоёв.

На данный момент это лучший одностадийный алгоритм, используется по умолчанию во всех реализованных многостадийных алгоритмах. Имеет наивысший потенциал для улучшения качества модели при обрезке. В опытах показал лучшие результаты при обрезке resnet50 на 50%.

#### Гиппер параметры
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training:  - До обучить после обрезки

## Мгогостадийные алгоритмы обрезки
[my pruning](#my-pruning), [AGP](#agp), [Linear](#linear), [LotteryTicket](#lotteryticket) - Алгоритмы, постепенно обрезающие модель одном из одностадийных или мгновенных алгоритмов прунинга. В ходе опытов выяснилось что лучшим вспомогательным алгоритмом для них является [TaylorFOWeight](#taylorfoweight), он же и применяется для обрезки в этой реализации.

#### Гиппер параметры
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training:  - До обучить после обрезки
- total_iteration:  - Сколько итераций обрезки делать.

### AGP
AGP - многостадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#agp-pruner) и доработан, для поддержки авто генерируемых сетей.

Разреженность s изменяется по формуле $$ s_{i+1} = s_k + (s_0 - s_k)(1 - \frac{i - i_0}{n \Delta i})^3 $$

### Linear
Linear - многостадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#nni.compression.pytorch.pruning.LinearPruner) и доработан, для поддержки авто генерируемых сетей.

Разреженность изменяется линейно [0, 0.1, 0.2, 0.3, 0.4, 0.5].

### LotteryTicket
LotteryTicket - многостадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#nni.compression.pytorch.pruning.LotteryTicketPruner) и доработан, для поддержки авто генерируемых сетей.

Каждая итерация сокращает $$ 1-(1-P)^{(\frac{1}{n})} $$ весов, оставшихся после предыдущей итерации, Где P - конечная разреженность

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
            
