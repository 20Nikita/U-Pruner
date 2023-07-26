# Pruning

## Description
Инструмент обрезки моделей.

Реализованы алгоритмы: [my pruning](#my-pruning), [L2Norm](#l2norm), [FPGM](#fpgm), [TaylorFOWeight](#taylorfoweight), [AGP](#agp), [Linear](#linear), [LotteryTicket](#lotteryticket).
## my pruning
my_pruning - алгоритм обрезки, основанный на [NetAdapt](https://arxiv.org/abs/1804.03230). Разработан специально для генерируемых сетей алгоритмами NAS. Занимает много времени, но работает одинаково хорошо с любой архитектурой сети и степенью сжатия.

Алгоритм разбивает сеть на блоки c помочью model.named_parameters(), обрезает каждый блок низкоуровневым алгоритмом TaylorFOWeight или L2Norm, до обучает и выбирает лучший по accuracy. Так по 1 блоку постепенно обрезает всю сеть. 

Поддерживается обрезка torch.nn.modules.conv.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.linear.Linear.

#### Гиперпараметры
- alf - Сохранить кратность первых двух каналов конволюции числу alf (33 * 65 * 7 * 7)->(32 * 64 * 7 * 7)
- P - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- cart - На каких картах обучать [0,0,1,1,1,2] (для параллельной работы, запустится обрезка 6 блоков с тренировкой на соответствующих картах)
- iskl - Название слоёв, которые не нужно обрезать [input_stem.0.conv,blocks.1.conv2.conv]
- algoritm - TaylorFOWeight, L2Norm. Прунинг в основе. TaylorFOWeight в 2 раза дольче L2Norm но немного лучше.
- resize_alf - Обрезать всю сеть до кратности alf (Если False то могут остаться не кратные свертки при условии что их обрезка сильно портит качество)
- delta_crop - Сколько % от блока резать за 1 итерацию 0.1->(100 * 100 * 7 * 7)->(90 * 100 * 7 * 7)
- start_iteration - Итерация, с которой продолжить работу. Влияет только на файл с логами
- load - Позволяет продолжить работу после ошибки

## Мгновенные алгоритмы обрезки
[L2Norm](#l2norm), [FPGM](#fpgm) - мгновенные алгоритмы обрезки. Идеальный вариант если нужно отрезать до 10% от сети.

#### Гиперпараметры
- P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training: - До обучить после обрезки

### L2Norm
L2Norm - мгновенный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#l1-norm-pruner) и доработан, для поддержки авто генерируемых сетей.

Вычисляет L2 норму в 1 измерении сверточных слоёв и удаляет строки с наименьшим значением этой метрики.

Особенно хорошо справляется с архитектурами подобными mobilenetv2

### FPGM
FPGM - мгновенный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#fpgm-pruner) и доработан, для поддержки авто генерируемых сетей.

Вычисляет медиану в 1 измерении сверточных слоёв и удаляет строки с наименьшим значением этой метрики.

Особенно хорошо справляется с архитектурами подобными resnet

## Одностадийные алгоритмы обрезки
Одностадийные алгоритмы обрезки нуждаются в информации об инференсе модели. Требуется пройти 1 эпоху обучения для сбора данных перед обрезкой, но благодаря этому алгоритмы имеют гораздо больший потенциал.

#### Гиперпараметры
- P - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training: - До обучить после обрезки

### TaylorFOWeight
TaylorFOWeight - одностадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#nni.compression.pytorch.pruning.TaylorFOWeightPruner) и доработан, для поддержки авто генерируемых сетей.

Работает на основе оценочной важности, рассчитанной из разложения Тейлора первого порядка для 1 измерении весов сверточных слоёв.

На данный момент это лучший одностадийный алгоритм, используется по умолчанию во всех реализованных многостадийных алгоритмах. Имеет наивысший потенциал для улучшения качества модели при обрезке. В опытах показал лучшие результаты при обрезке resnet50 на 50%.

#### Гиперпараметры
- P - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training: - До обучить после обрезки

## Многостадийные алгоритмы обрезки
[my pruning](#my-pruning), [AGP](#agp), [Linear](#linear), [LotteryTicket](#lotteryticket) - Алгоритмы, постепенно обрезающие модель одном из одностадийных или мгновенных алгоритмов прунинга. В ходе опытов выяснилось, что лучшим вспомогательным алгоритмом для них является [TaylorFOWeight](#taylorfoweight), он же и применяется для обрезки в этой реализации.

#### Гиппер параметры
- P - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
- training: - До обучить после обрезки
- total_iteration: - Сколько итераций обрезки делать.

### AGP
AGP - многостадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#agp-pruner) и доработан, для поддержки авто генерируемых сетей.

Разреженность s изменяется по формуле $` s_{i+1} = s_k + (s_0 - s_k)(1 - \frac{i - i_0}{n \Delta i})^3 `$

### Linear
Linear - многостадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#nni.compression.pytorch.pruning.LinearPruner) и доработан, для поддержки авто генерируемых сетей.

Разреженность изменяется линейно [0, 0.1, 0.2, 0.3, 0.4, 0.5].

### LotteryTicket
LotteryTicket - многостадийный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#nni.compression.pytorch.pruning.LotteryTicketPruner) и доработан, для поддержки авто генерируемых сетей.

Каждая итерация сокращает $` 1-(1-P)^{(\frac{1}{n})} `$ весов, оставшихся после предыдущей итерации, Где P - конечная разреженность

## Example
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
            
## Рекомендации

### Загрузка модели
Для загрузки обрезанной модели нужно в interfaces/interfaces/tools.py заменить

    def build_net(
        interface: OrderedDict = OrderedDict(), 
        pretrained: bool = False, 
        num_classes: int = None,
    ) -> mc.Module:
        algname = interface['nas']['algname']
        if algname == 'personal':
            net = build_personal(interface=interface, pretrained=pretrained, num_classes=num_classes)
        elif algname == 'ofa':
            net = build_ofa(interface=interface, pretrained=pretrained, num_classes=num_classes)
        else:
            raise Exception(f'This {algname} model is not yet used in this repository')
        return net
на

    def build_pruning(
        interface: OrderedDict = OrderedDict(), 
        pretrained: bool = False, 
        net:  mc.Module = None,
    ) -> mc.Module:
        model = net
        resize = interface['pruning']['summary']['resize']
        for component in resize:
            layer_name = component['name'].split(".weight")[0].split(".bias")[0]
            # print(component)
            if component['type'] == 'torch.nn.modules.conv.Conv2d':
                in_channels = component['shape'][0]
                out_channels = component['shape'][1]
                groups = eval("model.{}".format(layer_name)).groups
                groups = 1 if groups == 1 else in_channels
                kernel_size = eval("model.{}".format(layer_name)).kernel_size
                stride = eval("model.{}".format(layer_name)).stride
                padding = eval("model.{}".format(layer_name)).padding
                dilation = eval("model.{}".format(layer_name)).dilation
                padding_mode = eval("model.{}".format(layer_name)).padding_mode
                bias = eval("model.{}".format(layer_name)).bias
                new_pam = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, 
                                          stride = stride, padding = padding, dilation = dilation, groups = groups, 
                                          bias = bias, padding_mode = padding_mode)
                exec("model.{} = new_pam".format(layer_name))
            elif component['type'] == 'torch.nn.modules.batchnorm.BatchNorm2d':
                num_features = component['shape'][0]
                eps = eval("model.{}".format(layer_name)).eps
                momentum = eval("model.{}".format(layer_name)).momentum
                affine = eval("model.{}".format(layer_name)).affine
                track_running_stats = eval("model.{}".format(layer_name)).track_running_stats
                new_pam = torch.nn.BatchNorm2d(num_features, eps = eps, momentum = momentum, affine = affine, 
                                               track_running_stats = track_running_stats)
                exec("model.{} = new_pam".format(layer_name))
            elif component['type'] == 'torch.nn.modules.linear.Linear':
                in_features = component['shape'][0]
                out_features = component['shape'][1]
                new_pam = torch.nn.Linear(in_features, out_features)
                exec("model.{} = new_pam".format(layer_name))
        if pretrained:
            params = interface['params']
            model.load_state_dict(params)
        return model

    def build_net(
        interface: OrderedDict = OrderedDict(), 
        pretrained: bool = False, 
        num_classes: int = None,
    ) -> mc.Module:
        is_pruning = False
        if ('pruning' in interface) and interface['pruning']['is_pruning'] == True:
            is_pruning = True
        algname = interface['nas']['algname']
        if algname == 'personal':
            net = build_personal(interface=interface, pretrained=(pretrained and not is_pruning), num_classes=num_classes)
        elif algname == 'ofa':
            net = build_ofa(interface=interface, pretrained=(pretrained and not is_pruning), num_classes=num_classes)
        else:
            raise Exception(f'This {algname} model is not yet used in this repository')
        if is_pruning:
            net = build_pruning(interface=interface, pretrained=pretrained, net=net)
        return net
    
или перейти в ветку gordeev_dev в interfaces при сборе контейнера

### Варианты применения
Алгоритмы рассчитаны не на формирование архитектуры, а на удержание качества во время уменьшения модели. Поэтому даже если планируется учить модель с нуля после прунинга ОБЯЗАТЕЛЬНО загружать предобученные веса.


#### Золотой молоток
Алгоритм [my pruning](#my-pruning) выдаст лучшее качество в любой ситуации, но за универсальность платишь временем. В отличие от других многостадийных алгоритмов my pruning  обрезает за каждую итерацию не предсказуемый процент от сети. Число итераций не ограниченно. Поэтому точно оценить время окончания работы невозможно.

##### Рекомендуемые гиперпараметры
    my_pruning:
        alf: 32                       # Сохранить кратность первых двух каналов конволюции числу alf (33*65*7*7)->(32*64*7*7)
        P: 0.8                        # Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
        cart: [6,7]                   # На каких картах обучать [0,0,1,1,1,2] (для параллельного обучения, запустится 6 независимых тренировок на соответствующих картах)
        iskl: []                      # Название слоёв, которые не нужно обрезать [input_stem.0.conv,blocks.1.conv2.conv]
        algoritm: TaylorFOWeight      # TaylorFOWeight, L2Norm. Прунинг в основе.TaylorFOWeight в 2 раза дольще L2Norm но немного лучше.
        resize_alf: True              # Обрезать всю сеть до кратности alf (Если False то могут остаться не кратные свертки при условии что их обрезка сильно портит качество)
        delta_crop: 0.1               # Сколько % от текущей свертки резать за 1 итерацию 0.1->(100*100*7*7)->(90->100*7*7)
        restart:
            start_iteration: 0                                      # Итерация с которой продолжить работу (default: 0)
            load: /workspace/snp/elbrus_chest_mbnet/orig_model.pth  # Путь к модели (default: exp_save + "/" + model_name + "/" + "orig_model.pth")
    retraining:
        num_epochs: 1
        lr: 0.00001
    training:
        num_epochs: 10
        lr: 0.00001

Для модели elbrus_chest_mbnet

    constraint: 33.333333333333336
    constraint_type: latency
    optimize_val: latency
    pred_acc: 100.39780139923096
    pred_optimized_val: 33.306026458740234
    test_top1: 0.9659090909090909
    val_top1: 0.9024390243902439
Результат будет таким

    'size': 0.1970215054204825,
    'val_accuracy': 0.9563218355178833,
    'time': '20h 20m 48s'}
                            
#### Мгновенный результат
Алгоритмы [L2Norm](#l2norm) и [FPGM](#fpgm) отработают за несколько секунд. Если архитектура больше похожа на mobilenetv2 лучше использовать [L2Norm](#l2norm). С resnet больше подойдет [FPGM](#fpgm). Однако модель будет выдавать результат как необученная архитектура. Требуется дообучить с маленьким lr или установить training: True.

##### Рекомендуемые гиперпараметры

    nni_pruning:
        P: 0.1                         # Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
        training: False                # Дообучить после обрезки
        gpu: 0                         # Карта для обучения

[TaylorFOWeight](#taylorfoweight) имеет куда больший потенциал если требуется быстро обрезать сеть не на крошечный процент. Перед обрезкой будет пройдена 1 холостая эпоха обучения, однако результат того стоит. 
        
#### Контролируемая по времени обрезка

Время работы алгоритмов [AGP](#agp), [Linear](#linear), [LotteryTicket](#lotteryticket) составит t * (training.num_epochs + 1) * total_iteration, где t - время 1 эпохи обучения.

При  небольшой обрезке лучше использовать [Linear](#linear), на средней [LotteryTicket](#lotteryticket), а при экстремальной [AGP](#agp).

##### Рекомендуемые гиперпараметры

    nni_pruning:
        P: 0.8                         # Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
        training: True                 # Дообучить после обрезки
        total_iteration: 100           # Итерации обрезки, используется в алгоритмах AGP, Linear, LotteryTicket.
        gpu: 0                         # Карта для обучения
        
    training:
        num_epochs: 100
        lr: 0.00001
        
Для модели elbrus_chest_mbnet

    constraint: 33.333333333333336
    constraint_type: latency
    optimize_val: latency
    pred_acc: 100.39780139923096
    pred_optimized_val: 33.306026458740234
    test_top1: 0.9659090909090909
    val_top1: 0.9024390243902439
Результат будет таким

    'size': 0.20194015798119477, 
    'val_accuracy': 0.9477011561393738,
    'time': '3h 49m 9s'
    
Если далее запустить my_pruning с P: 0, resize_alf: True (остальные гипперпараметры совпадают с рекомендуемыми для my_pruning).

Результат будет таким

    'size': 0.8869792831894044, 
    'val_accuracy': 0.9465517401695251, 
    'time': '0h 21m 25s'

## Применение для библиотечных моделей
Если модель в библиотеке реализованна 'просто' и без лишних функций, то достаточно добавить библиотеку в  requirements.txt и запустить прунинг с соответствующим конфигом

Если это ни так, требуется проделать следующее
1) Переписать/скопировать модель из библиотеки в файлы с расширением .py. На этом этапе важно получить 'точку входа' формирования модели. Все зависимые для неё компоненты должны быть в директории не выше чем точка входа.

2) Из модели требуется удалить всю логику не связанную с формированием модели: 
    - постобработку и предобработку
    - макросы (@compatibility(is_backward_compatible=True))
    - функции загрузки весов
    - символьную трассировку (if h % output_stride != 0 or w % output_stride != 0:)
    - вункции принимающие не очивидные оргументы (forward_call(*input, **kwargs))
3) Точку входа требуется сделать наследником torch.nn
и все компоненты сети с весами сети требуется брать из torch.nn или наследовать от torch.nn.

4) В той же директории, где находится точка входа нужно:
    - создать модельку как класс из точки входа
    - загрузить в неё виса оригенальной модельки
    - сохранить модельку в текущей директории с расширениеи .pth

Пример

    import my_point_of_entry
    model = my_point_of_entry.model()
    model.load_state_dict(weights_origin_model, strict=True)
    torch.save(model, "my_model.pth")

5) В Pruning.yaml установить
```
model:
    type_save_load: pth
    path_to_resurs: /workspace/proj/model #путь до каталога с точкой фхода и my_model.pth
    name_resurs: my_model # название сохраненной pth модели без расширения .pth
```
