# Pruning

### Description
Инструмент обрезки моделей.
Реализованны аггоритмы: [my_pruning](#my_pruning)

### my_pruning
my_pruning - алгоритм обрезки, основанный на [NetAdapt](https://arxiv.org/abs/1804.03230).
Адгоритм разбивает сеть на блоки c помошью model.named_parameters(), обрезает каджый блок нискоуровневым алгоритмом TaylorFOWeight или L2Norm, дообучает и выбирает лучший по Acc. Так по 1 блоку постепенно обрезает всю сеть. Поддержевается обрезка torch.nn.modules.conv.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.linear.Linear.

#### Гиппер параметры
alf        - Сохранить кратность первых двух коналов конволюции числу alf (33*65*7*7)->(32*64*7*7)
P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
cart       - На каких картах обучать [0,0,1,1,1,2] (для паралельной работы, запустится обрезка 6 блоков с тренировкой на соответствующих картах)
iskl       - Название слоёв, которые не нужно обрезать [input_stem.0.conv,blocks.1.conv2.conv]
algoritm   - TaylorFOWeight, L2Norm. Прунинг в основе.TaylorFOWeight в 2 раза дольще L2Norm но немного лучше.
resize_alf - Обрезать всю сеть до кратности alf (Если False то могут остатся не кратные свертки при условии что их обрезка сильно портит качество)
delta_crop - Сколько % от блока резать за 1 итерацию 0.1->(100*100*7*7)->(90->100*7*7)

### L2Norm
L2Norm - мгновенный алгоритм обрезки. Основан на реализации [nni](https://nni.readthedocs.io/en/stable/reference/compression/pruner.html#l1-norm-pruner) и доработан, для поддержки авогенерируемых сетей.
#### Гиппер параметры
P          - Сколько отрезать от сети (Пока основывается на ptflops.get_model_complexity_info)
training:  - Дообучить после обрезки