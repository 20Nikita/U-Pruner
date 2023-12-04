import argparse
import torch
import os
from ptflops import get_model_complexity_info
import math

# import yaml

# # Получить данные об обучаемых параметрах сети
# def get_type(module):
#     type_module = str(type(module))
#     if (
#         type_module == "<class 'torch.nn.modules.conv.Conv2d'>"
#         or type_module == "<class 'models.core.conv.Conv2d'>"
#     ):
#         return "Conv2d"
#     elif (
#         type_module == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>"
#         or type_module == "<class 'models.core.norm.BatchNorm2d'>"
#     ):
#         return "BatchNorm2d"
#     elif type_module == "<class 'torch.nn.modules.linear.Linear'>":
#         return "Linear"
#     else:
#         return type_module


# Получить имена компонентов сети, имеющих параметры
def get_stract(model):
    stact = []
    for name, module in model.named_modules():
        n_parametrs = 0
        for _ in module.named_parameters():
            n_parametrs += 1
        if n_parametrs < 3 and n_parametrs > 0:
            if isinstance(module, torch.nn.Conv2d):
                stact.append(
                    [
                        name,
                        "Conv2d",
                        [module.in_channels, module.out_channels],
                        module.kernel_size[0] * module.kernel_size[1],
                    ]
                )
            elif isinstance(module, torch.nn.BatchNorm2d):
                stact.append([name, "BatchNorm2d", [module.num_features]])
            elif isinstance(module, torch.nn.Linear):
                stact.append(
                    [name, "Linear", [module.in_features, module.out_features]]
                )
            else:
                stact.append([name, str(type(module)), "******************"])
    return stact


### Функции для получения мески
# Получить следующие элементы в графе graf, после текущего tek_point
def getNextElements(tek_point, graf):
    rez = []
    for candidate in graf:
        args = candidate.args
        # Кастыль для операции cat, torch.fx воспренимает его аргументы как лист.
        if isinstance(args[0], torch.fx.immutable_collections.immutable_list):
            args = args[0]
        for arg in args:
            if arg == tek_point:
                rez.append(candidate)
                break
    return rez


# Получить элементы в графе graf, стоящие перед tek_point
def getPreviousElements(tek_point, graf):
    rez = []
    for candidate in tek_point.args:
        rez.append(candidate)
    return rez


# Проверка на то, что этот элемент конечный
def endPoint(tek_point, model):
    if tek_point.name == "output":
        return True
    if tek_point.op != "call_module":
        return False
    submodule = model.get_submodule(tek_point.target)
    if (
        isinstance(submodule, torch.nn.Conv2d)
        and submodule.groups == 1
        or isinstance(submodule, torch.nn.Linear)
    ):
        return True
    else:
        return False


# Получить список из названия элемента и с какой стороны его резать
def getTarget(tek_point, model, bool_niz):
    if tek_point.op == "call_module":
        submodule = model.get_submodule(tek_point.target)
        if bool_niz:
            if isinstance(submodule, torch.nn.Conv2d) and submodule.groups == 1:
                return [str(tek_point.target), 1]
            elif isinstance(submodule, torch.nn.Linear):
                return [str(tek_point.target), 0]
            elif isinstance(submodule, torch.nn.Conv2d) or isinstance(
                submodule, torch.nn.BatchNorm2d
            ):
                return [str(tek_point.target), 0]
        else:
            if isinstance(submodule, torch.nn.Conv2d) and submodule.groups == 1:
                return [str(tek_point.target), 0]
            elif isinstance(submodule, torch.nn.Linear):
                return [str(tek_point.target), 1]
            elif isinstance(submodule, torch.nn.Conv2d) or isinstance(
                submodule, torch.nn.BatchNorm2d
            ):
                return [str(tek_point.target), 0]


# обьединить листы
def add(spisok, rez_recurs):
    if rez_recurs != None:
        spisok += rez_recurs


# Вызов следующей рекурсии, учитывая направление движения
def Add(spisok, tek_point, graf, model, bool_niz, glob, stops, isprint):
    s = "\t"
    if bool_niz:
        for el in getNextElements(tek_point, graf):
            if isprint:
                print(
                    f"{s*glob}Вниз  el = {el}, tek_point = {tek_point}, elements = {getNextElements(tek_point,graf)}"
                )
            add(
                spisok,
                recurs(el, graf, model, bool_niz, tek_point, glob + 1, stops, isprint),
            )
    else:
        for el in getPreviousElements(tek_point, graf):
            if isprint:
                print(
                    f"{s*glob}Вверх el = {el}, tek_point = {tek_point}, elements = {getPreviousElements(tek_point,graf)}"
                )
            add(
                spisok,
                recurs(el, graf, model, bool_niz, tek_point, glob + 1, stops, isprint),
            )


# Удалить дублирующиеся элементы в листе
def deletDublicat(spisok):
    rezalt = []
    for i in spisok:
        if not i in rezalt:
            rezalt.append(i)
    return rezalt


# Рекурсивный обход модели для получения взаимосвязанных элементов
def recurs(
    tek_point,
    graf,
    model,
    bool_niz: bool = True,
    pred_point=None,
    glob=0,
    stops=[],
    isprint=False,
    self_pass=False,
):
    spisok = []
    s = "\t"
    if isprint:
        print(f"{s*glob}Name = {tek_point}, stops = {stops}")
    if glob > 30:
        return spisok
    if tek_point.name in stops:
        if isprint:
            print(f"{s*glob}STOP name = {tek_point.name}")
        return spisok
    if isinstance(tek_point, torch.fx.node.Node):
        if len(tek_point.args) > 1:
            for el in tek_point.args:
                if isprint:
                    print(
                        f"{s*glob}add: el = {el}, bool_niz = {bool_niz}, tek_point = {tek_point}, args: {tek_point.args}"
                    )
                if isinstance(el, torch.fx.node.Node) and pred_point != el:
                    if not endPoint(el, model):
                        if isprint:
                            print(
                                f"{s*glob}!!!, el = {el}, tek_point = {tek_point}, bool_niz = {bool_niz}"
                            )
                        add(
                            spisok,
                            recurs(
                                el,
                                graf,
                                model,
                                bool_niz,
                                tek_point,
                                glob + 1,
                                stops + [tek_point.name],
                                isprint,
                                True,
                            ),
                        )
                    if isprint:
                        print(
                            f"{s*glob}el = {el}, tek_point = {tek_point}, bool_niz = {not bool_niz}"
                        )
                    add(
                        spisok,
                        recurs(
                            el,
                            graf,
                            model,
                            not bool_niz,
                            tek_point,
                            glob + 1,
                            stops + [tek_point.name],
                            isprint,
                        ),
                    )
            if isprint:
                print(
                    f"{s*glob}add_prodol: tek_point = {tek_point}, bool_niz = {bool_niz}, args: {tek_point.args}"
                )
            Add(
                spisok,
                tek_point,
                graf,
                model,
                bool_niz,
                glob,
                stops + [tek_point.name],
                isprint,
            )
            spisok = deletDublicat(spisok)
            return spisok
        else:
            if endPoint(tek_point, model) and pred_point != None:
                target = getTarget(tek_point, model, bool_niz)
                if isprint:
                    print(f"{s*glob}End {target}")
                if target:
                    spisok.append(target)
                return spisok
            else:
                Add(spisok, tek_point, graf, model, bool_niz, glob, stops, isprint)
            if not self_pass:
                target = getTarget(tek_point, model, bool_niz)
                if target:
                    spisok.append(target)
            return spisok

class MyCustomTracer(torch.fx.Tracer):
    
    def __init__(
        self,
        autowrap_modules = (math,),
        autowrap_functions = (),
        param_shapes_constant = False,
        class_name = None
    ) -> None:
        super().__init__(autowrap_modules, autowrap_functions ,param_shapes_constant)
        self.class_name = class_name
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        class_names = self.class_name
        temp = False
        if class_names!= None:
            for class_name in class_names:
                temp =  temp or m.__module__.startswith(class_name)
        return (
            (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn") or temp)
            and not isinstance(m, torch.nn.Sequential)
        )
def symbolic_trace(root,concrete_args= None, class_name = None):
    tracer = MyCustomTracer(class_name = class_name)
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return torch.fx.GraphModule(tracer.root, graph, name)
# Получение маски взаимосвязанных элементов
def get_mask(model, isPrint1=False, isPrint2=False, class_name = None):
    # Получить граф модели
    gm = symbolic_trace(model, class_name = class_name)
    # gm = torch.fx.symbolic_trace(model)
    nodes = []  # Имена нод в формате torch.fx
    crops = []
    for n in gm.graph.nodes:
        if len(n.args) > 0:
            nodes.append(n)
    # Пройтись по fx нодам
    for i, n in enumerate(nodes):
        # Если у ноды есть взаимосвязанные элементы
        # и нода являестя торчевским компонентом с параметрами
        if len(n.args) > 0 and n.op == "call_module":
            submodule = model.get_submodule(n.target)
            # Если нода полноценная свертка или линейный слой
            if (
                isinstance(submodule, torch.nn.Conv2d)
                and submodule.groups == 1
                or isinstance(submodule, torch.nn.Linear)
            ):
                if isPrint1:
                    print(i, len(crops), n.target)
                m = recurs(n, nodes, model, isprint=isPrint2)[::-1]
                m[0][1] = 0
                crops.append(m)
    return crops


# Получить сумарное количество строк в конвалюционной матрице. Мера размера сети
def get_size(model, shape):
    macs, params = get_model_complexity_info(
        model,
        (3, shape[0], shape[1]),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return params


# Удалить веса. Вход: weight - torch.nn.Parameter (Веса),
# i - list (инексы строк массива весов, которые нужно удалить),
# j - list (инексы столбцов массива весов, которые нужно удалить)
def delet_weight(weight, i=-1, j=-1):
    # Удалить строки
    if i != -1:
        for k in range(len(i)):
            k = i[len(i) - 1 - k]
            weight = torch.nn.Parameter(torch.cat([weight[0:k], weight[k + 1 :]]))
    # Удалить столбцы
    if j != -1:
        # Замена индексов которые нужно удалить на индексы, которые нужно оставить
        j = set(range(len(weight[0]))) - set(j)
        # Запомнить размерность
        weight_shape = list(weight.shape)
        weight_shape[1] = 1  # Удаляем столбцы
        # Пересобрать веса из индексов, которые нужно оставить
        weight = [weight[:, k, :, :].reshape(weight_shape) for k in j]
        weight = torch.nn.Parameter(torch.cat(weight, 1))
    return weight


def delet_weight_Linear(weight, i=-1, j=-1):
    # Удалить строки
    if i != -1:
        for k in range(len(i)):
            k = i[len(i) - 1 - k]
            weight = torch.nn.Parameter(torch.cat([weight[0:k], weight[k + 1 :]]))
    # Удалить столбцы
    if j != -1:
        # Замена индексов которые нужно удалить на индексы, которые нужно оставить
        j = set(range(len(weight[0]))) - set(j)
        # Запомнить размерность
        weight_shape = list(weight.shape)
        weight_shape[1] = 1  # Удаляем столбцы
        # Пересобрать веса из индексов, которые нужно оставить
        weight = [weight[:, k].reshape(weight_shape) for k in j]
        weight = torch.nn.Parameter(torch.cat(weight, 1))
    return weight


# Замена слоя такимже, но с обрезанными параметрами и уменьшенным размером.
def delet(model, Delet_Name_sloi, i=-1, j=-1):
    # Узнать тип слоя
    # types = get_type(eval("model.{}".format(Delet_Name_sloi)))
    submodule = model.get_submodule(Delet_Name_sloi)
    new_pam = None
    if isinstance(submodule, torch.nn.Conv2d):
        groups = submodule.groups
        groups = 1 if groups == 1 else groups - len(i)
        # Если в квадратной матрице с 1 продублированным n раз столбцом мне нужно удалить столбец. Я удалю строку по индексу столбца.
        # Проблема в том, что её отображаемая размерность (a,b,c,d), а фактические веса (a,1,c,d) с groups = b
        if groups != 1 and i == -1:
            i = j
            j = -1

        # Изменить размер столбцов
        in_channels = submodule.in_channels
        in_channels = in_channels if j == -1 else in_channels - len(j)
        # Если groups != 1, то это квадратная матрица с продублированными столбцами, а в ней я удаляю строки по индексу столбца
        in_channels = in_channels if groups == 1 else in_channels - len(i)

        # Изменить размер строк
        out_channels = submodule.out_channels
        out_channels = out_channels if i == -1 else out_channels - len(i)

        # Эти параметры копирую без изменений
        kernel_size = submodule.kernel_size
        stride = submodule.stride
        padding = submodule.padding
        dilation = submodule.dilation
        padding_mode = submodule.padding_mode

        # Получить обрезанные веса
        weight = delet_weight(submodule.weight, i, j)
        # bias - обучаемый параметр, размерностью 1, с количеством элементов = количеству строк в конвалюционном слое. Но его может не быть.
        bias = submodule.bias
        bias_w = bias
        if bias != None:
            bias = True
            if i != -1:
                bias_w = delet_weight(submodule.bias, i, j)

        # Создать укопию слоя с обрезанными параметрами, но рандомными весами
        new_pam = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        # Передать веса
        new_pam.weight = weight
        new_pam.bias = bias_w
        # print(new_pam,new_pam.weight.shape)

        # Заменить слой обрезанным
        # exec("model.{} = new_pam".format(Delet_Name_sloi))
    if isinstance(submodule, torch.nn.BatchNorm2d):
        # Изменить размер строк
        num_features = submodule.num_features - len(i)

        # Эти параметры копирую без изменений
        eps = submodule.eps
        momentum = submodule.momentum
        affine = submodule.affine
        track_running_stats = submodule.track_running_stats

        # Создать укопию слоя с обрезанными параметрами, но рандомными весами
        new_pam = torch.nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        # Передать веса
        new_pam.weight = delet_weight(submodule.weight, i, j)
        new_pam.bias = delet_weight(submodule.bias, i, j)

        # Заменить слой обрезанным
        # exec("model.{} = new_pam".format(Delet_Name_sloi))
    if isinstance(submodule, torch.nn.Linear):

        in_features = submodule.in_features
        in_features = in_features if i == -1 else in_features - len(i)

        out_features = submodule.out_features
        out_features = out_features if j == -1 else out_features - len(j)

        bias = submodule.bias

        new_pam = torch.nn.Linear(in_features, out_features)

        # Передать веса
        new_pam.weight = delet_weight_Linear(submodule.weight, j, i)
        new_pam.bias = delet_weight_Linear(submodule.bias, j)

        # exec("model.{} = new_pam".format(Delet_Name_sloi))
    # Заменить слой обрезанным
    sloi = Delet_Name_sloi.split(".")
    t = model
    for s in sloi[:-1]:
        t = t.__dict__["_modules"][s]
    t.__dict__["_modules"][sloi[-1]] = new_pam


# Заменить строку формата *.n.*.m.* на *[n].*[m].*
# def rename(name):
#     new_name = ""
#     name_split = name.split(".")
#     for i in range(len(name_split)):
#         try:
#             new_name = new_name + "[{}]".format(int(name_split[i]))
#         except:
#             new_name = new_name + ".{}".format(name_split[i])
#     try:
#         t = int(name_split[0])
#         return new_name
#     except:
#         return new_name[1:]


# Инвентировать список ([a,b,c] => [c,b,a])
def obr(names):
    names_invert = names.copy()
    for i in range(len(names)):
        names_invert[len(names) - i - 1] = names[i]
    return names_invert


# Конвертация маски nni в индексы удоляемых строк
def get_delet(masks):
    delet = []
    for i in range(len(masks)):
        if masks[i][0][0][0] == 0:
            delet.append(i)
    return delet


# Пробегает по переданным параметрам модели (вперед или назад определяется инвертированным масивов параметров),
# находит слой который нужно удалить Delet_Name_sloi и удаляет в нем и следуюхих строки Delet_indeks.
# Останавливается, когда дошел до конвалюционного слоя с groups == 1 и в нем удаляет столбец
# при движении вперед и строку при движении назад. Направление знает по параметру obratno
# def go_model(model, names, Delet_Name_sloi, Delet_indeks, obratno=False, priint=False):
#     is_delett = False  # В режиме удаления
#     for name in names:  # Пройтись по слоям модели с параметрами
#         Name_the_sloi = rename(name[0].split(".weight")[0])  # Название текущего слоя
#         not_bias = len(name[0].split(".bias")) == 1  # Чтобы не порезать дважды 1 слой
#         if (
#             Delet_Name_sloi == name[0].split(".weight")[0]
#         ):  # Нашли слой, который нужно удалить
#             is_delett = True
#             if obratno:
#                 continue  # Текущий сдлй уже был порезан при движении вперед
#         # Если текущий стой - слой конвалюции и мы в режиме удаления, и этот параметр не bias
#         elif (
#             get_type(eval("model.{}".format(Name_the_sloi))) == "Conv2d"
#             and is_delett
#             and not_bias
#         ):
#             if (
#                 eval("model.{}".format(Name_the_sloi)).groups == 1
#             ):  # Этот слой не состоит из дубликатов столбцов
#                 is_delett = False  # Выходим из режима удаления
#                 if not priint:  # Отображаем или удаляем, в зависимости от режима работы
#                     if obratno:
#                         delet(model, Name_the_sloi, i=Delet_indeks, j=-1)
#                     else:
#                         delet(model, Name_the_sloi, i=-1, j=Delet_indeks)
#                 else:
#                     print("K", name)
#         if (
#             is_delett and not_bias
#         ):  # Удаляем, если мы в режиме удаления и этот параметр не bias
#             if not priint:
#                 delet(model, Name_the_sloi, i=Delet_indeks, j=-1)
#             else:
#                 print("S", name)  #  Или принтуем, если таков режим работы


# Пробегает по переданным параметрам модели (вперед или назад определяется инвертированным масивов параметров),
# находит слой который нужно удалить Delet_Name_sloi, запоминает его размерность
# и удаляет в нем и всех следуюхих слоях с этой размерностью строки и столбцы Delet_indeks.
# Останавливается, если stop = True и когда дошел до конвалюционного слоя из другово блока
# при движении вперед удаляем в нем столбец а при движении назад ничего. Направление знает по параметру obratno
# def go_model_ofa_scip(
#     model, names, Delet_Name_sloi, Delet_indeks, obratno=False, priint=False, stop=True
# ):
#     is_delett = False  # В режиме удаления
#     Rename_Delet_Name_sloi = rename(Delet_Name_sloi)  # Обращение к слою
#     size = -1
#     for name in names:  # Пройтись по слоям модели с параметрами
#         not_bias = len(name[0].split(".bias")) == 1  # Чтобы не порезать дважды 1 слой
#         Name_the_sloi = rename(name[0].split(".weight")[0])  # Название текущего слоя
#         if (
#             Delet_Name_sloi == name[0].split(".weight")[0]
#         ):  # Нашли слой, который нужно удалить
#             is_delett = True  # Перейти в режим удаления
#             size = name[2][-1]  # Запомнить размерность удоляемого слоя
#             if obratno:  # Если мы удаляем двигаясь назад по сети
#                 size = size + len(
#                     Delet_indeks
#                 )  # востонавливаем информацию о размерности
#                 continue  # ведь текущий слой уже порезан
#         elif stop:  # Мы удаляем по закономерностям офы
#             # Это другой блок и мы в режиме удаления
#             if (
#                 is_delett
#                 and Rename_Delet_Name_sloi.split("[")[1].split("]")[0]
#                 != Name_the_sloi.split("[")[1].split("]")[0]
#             ):
#                 if obratno:  # При движении назад
#                     is_delett = (
#                         False  # удалять последний элемент блока в конце не нужно
#                     )
#                     break
#                 elif (
#                     name[1] == "Conv2d"
#                 ):  # При движении вперед, если этот слой - слой конволюции
#                     is_delett = False  # Выходим из режима удаления
#                     if name[2][-1] == size or (
#                         name[2][0] == size and name[1] == "Linear"
#                     ):  # Если у первого элемента следующего слоя размерность как у удоляемого
#                         if (
#                             not priint
#                         ):  # Удоляем или принтуем в зависимости от режима работы
#                             if obratno:
#                                 print("K", name)
#                                 delet(model, Name_the_sloi, i=Delet_indeks, j=-1)
#                             else:
#                                 print("K", name)
#                                 delet(model, Name_the_sloi, i=-1, j=Delet_indeks)
#                         else:
#                             print("K", name)
#         if is_delett and not_bias:  # Если мы в режиме удаления и этот параметр не bias
#             if name[2][-1] == size or (
#                 name[2][0] == size and name[1] == "Linear"
#             ):  # Размерность строк совпадает (конволюции и бачнорм) с обзезаемым слоем
#                 if not priint:  # Удоляем их если мы в режиме удаления
#                     print("S", name)
#                     delet(model, Name_the_sloi, i=Delet_indeks, j=-1)
#                 else:
#                     print("S", name)
#             if (
#                 name[2][0] == size and name[1] == "Conv2d"
#             ):  # Столбцы конволюционных матриц с этой размерностью
#                 if (
#                     eval("model.{}".format(Name_the_sloi)).groups == 1
#                 ):  # которые не состоят из дублекатов себя
#                     if not priint:  # ттоже удаляем или принтуем
#                         print("S", name)
#                         delet(model, Name_the_sloi, i=-1, j=Delet_indeks)
#                     else:
#                         print("S", name)


# Проходет по маске и запускает удаление строк слоя из маски и связанных с ним слоём
# def compres(model, masks):
#     for i in masks:
#         for j in masks[i]:

#             Name = rename(i)  # Обращение к слою
#             Delet_indeks = get_delet(
#                 masks[i][j]
#             )  # получить индексы строк для удаления из маски параметров
#             # Удалить компоненты слоя по маске
#             go_model(model, get_stract(model), i, Delet_indeks, obratno=False)
#             # Если это квадратная матрица
#             if (
#                 eval("model.{}".format(Name)).in_channels
#                 == eval("model.{}".format(Name)).out_channels
#             ):
#                 # То удалять и в обратную сторону
#                 go_model(model, obr(get_stract(model)), i, Delet_indeks, obratno=True)
#             # Повторить но только отображать какие слои резалить
#             go_model(
#                 model, get_stract(model), i, Delet_indeks, obratno=False, priint=True
#             )
#             if (
#                 eval("model.{}".format(Name)).in_channels
#                 == eval("model.{}".format(Name)).out_channels
#             ):
#                 go_model(
#                     model,
#                     obr(get_stract(model)),
#                     i,
#                     Delet_indeks,
#                     obratno=True,
#                     priint=True,
#                 )


# # Проходет по маске и запускает удаление строк слоя из маски и связанных с ним слоём
# def compres2(model, masks, ofa=False, stop=True):
#     for i in masks:
#         for j in masks[i]:
#             Name = rename(i)  # Обращение к слою
#             print("d:", Name)
#             # Удаление стандартной модели, по типу mobilenetv2
#             if not ofa:
#                 Delet_indeks = get_delet(masks[i][j])
#                 go_model(model, get_stract(model), i, Delet_indeks, obratno=False)
#                 if (
#                     eval("model.{}".format(Name)).in_channels
#                     == eval("model.{}".format(Name)).out_channels
#                 ):
#                     go_model(
#                         model, obr(get_stract(model)), i, Delet_indeks, obratno=True
#                     )
#                 go_model(
#                     model,
#                     get_stract(model),
#                     i,
#                     Delet_indeks,
#                     obratno=False,
#                     priint=True,
#                 )
#                 if (
#                     eval("model.{}".format(Name)).in_channels
#                     == eval("model.{}".format(Name)).out_channels
#                 ):
#                     go_model(
#                         model,
#                         obr(get_stract(model)),
#                         i,
#                         Delet_indeks,
#                         obratno=True,
#                         priint=True,
#                     )
#             # Удаление модели, с особенностями ofa
#             elif stop:
#                 Delet_indeks = get_delet(masks[i][j])
#                 go_model_ofa_scip(
#                     model, get_stract(model), i, Delet_indeks, obratno=False
#                 )
#                 go_model_ofa_scip(
#                     model, obr(get_stract(model)), i, Delet_indeks, obratno=True
#                 )
#                 go_model_ofa_scip(
#                     model,
#                     get_stract(model),
#                     i,
#                     Delet_indeks,
#                     obratno=False,
#                     priint=True,
#                 )
#                 go_model_ofa_scip(
#                     model,
#                     obr(get_stract(model)),
#                     i,
#                     Delet_indeks,
#                     obratno=True,
#                     priint=True,
#                 )
#             # 100% работающее удаление слоев модели, если асобенности слишком тяжолые
#             else:
#                 Delet_indeks = get_delet(masks[i][j])
#                 go_model_ofa_scip(
#                     model, get_stract(model), i, Delet_indeks, obratno=False, stop=False
#                 )
#                 go_model_ofa_scip(
#                     model,
#                     obr(get_stract(model)),
#                     i,
#                     Delet_indeks,
#                     obratno=True,
#                     stop=False,
#                 )
#                 go_model_ofa_scip(
#                     model,
#                     get_stract(model),
#                     i,
#                     Delet_indeks,
#                     obratno=False,
#                     priint=True,
#                     stop=False,
#                 )
#                 go_model_ofa_scip(
#                     model,
#                     obr(get_stract(model)),
#                     i,
#                     Delet_indeks,
#                     obratno=True,
#                     priint=True,
#                     stop=False,
#                 )


# Проходет по маске и запускает удаление строк слоя из маски и связанных с ним слоём
def compres3(model, masks, sours_mask):
    for Name in masks:
        for j in masks[Name]:
            Delet_indeks = get_delet(
                masks[Name][j]
            )  # получить индексы строк для удаления из маски параметров
            # Удалить компоненты слоя по маске
            for mask in sours_mask:
                if mask[0][0] == Name:
                    for k, (sloi, i) in enumerate(mask):
                        if i == 0:
                            delet(model, sloi, i=Delet_indeks)
                        elif i == 1:
                            delet(model, sloi, j=Delet_indeks)


# Получить переданные параметры
def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self_ind",
        type=int,
        default=0,
        help="Порядковый номер работника. Чтобы отличать их между собой",
    )
    parser.add_argument("--cuda", type=int, default=0, help="Номер карты")
    parser.add_argument(
        "--load",
        type=str,
        help='Путь к модели в формате совместимым с "model = torch.load(load)"',
    )
    parser.add_argument("--name_sloi", type=str, help="Название слоя для удаления")
    parser.add_argument(
        "--sparsity", type=float, default=0.1, help="% который нужно обрезать от слоя"
    )
    parser.add_argument("--orig_size", type=float, help="Размер этого слоя до обрезки")
    parser.add_argument(
        "--iterr",
        type=int,
        default=0,
        help="Итерация алгоритма обрезки, нужен для логов",
    )
    parser.add_argument(
        "--algoritm",
        type=str,
        default="TaylorFOWeight",
        help="Низкоуровневый алгоритм обрезки. Выбор из: TaylorFOWeight, L2Norm",
    )
    parser.add_argument("--config", type=str, help="Путь до конфигурационного файла")
    param = parser.parse_args()
    return param


# Часть алгоритма вынесенная в ф-ю, чтобы не дублировать код.
# Понять что прунинг неудался можно только запустив обучение и увидив ошибку.
# Запуск прунинга и последующее дообучение с записью логов в файл.
# def pruning_type(model, masks, do, param, config_list, config, type_pruning="defolt"):
#     import yaml
#     import torch.optim as optim
#     import training as trainer

#     alf = config.my_pruning.alf
#     lr = config.retraining.lr
#     num_epochs = config.retraining.num_epochs
#     snp = config.path.exp_save + "/" + config.path.modelName
#     modelName = config.path.modelName
#     fil_it = snp + "/" + modelName + "_it{}.txt".format(param.iterr)
#     model = torch.load(param.load)  # Загрузка модели (модель осталась порезанной)
#     # Кастыль для сегментации в офе
#     if config.model.type_save_load == "interface":
#         model.backbone_hooks._attach_hooks()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     print(type_pruning)
#     if do[0] * do[1] == param.orig_size:
#         if type_pruning == "defolt":
#             compres2(model, masks)  # Запуск прунинга
#         elif type_pruning == "ofa":
#             compres2(model, masks, ofa=True)  # Запуск прунинга
#         elif type_pruning == "total":
#             compres2(model, masks, ofa=True, stop=False)  # Запуск прунинга

#     model.to(device)  # Замененные слои не на карте
#     # Узнать размер сверток после прунинга
#     for name in get_stract(model):
#         if name[0].split(".weight")[0] == config_list[0]["op_names"][0]:
#             posle = name[2]
#             break

#     # После обрезки сеть обрезслась и кратна alf

#     if do != posle and posle[1] % alf == 0:
#         # Дообучаем
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         model, loss, acc, st, time_elapsed2 = trainer.retrainer(
#             model,
#             optimizer,
#             trainer.criterion,
#             num_epochs=num_epochs,
#             ind=param.self_ind,
#         )
#         # Запись результата
#         f = open(fil_it, "a")
#         strok = (
#             str(param.self_ind)
#             + " "
#             + config_list[0]["op_names"][0]
#             + " "
#             + str(acc)
#             + " "
#             + str(do)
#             + " "
#             + str(posle)
#             + " "
#             + "type_pruning "
#             + type_pruning
#             + "\n"
#         )
#         f.write(strok)
#         f.close()
#         # Сохранение модели
#         # Кастыль для сегментации в офе
#         if config.model.type_save_load == "interface":
#             model.backbone_hooks._clear_hooks()
#         torch.save(
#             model,
#             snp
#             + "/"
#             + modelName
#             + "_"
#             + config_list[0]["op_names"][0]
#             + "_it_{}_acc_{:.3f}.pth".format(param.iterr, acc),
#         )
#     else:
#         # Запись размерностей до и после прунинга
#         f = open(fil_it, "a")
#         strok = (
#             str(param.self_ind)
#             + " "
#             + config_list[0]["op_names"][0]
#             + " "
#             + "EROR"
#             + " "
#             + str(do)
#             + " "
#             + str(posle)
#             + " "
#             + "type_pruning "
#             + type_pruning
#             + "\n"
#         )
#         f.write(strok)
#         f.close()


def pruning_mask(model, masks, do, param, config_list, config):
    import yaml
    import torch.optim as optim
    import training as trainer
    import json

    alf = config.my_pruning.alf
    lr = config.retraining.lr
    num_epochs = config.retraining.num_epochs
    snp = os.path.join(config.path.exp_save, config.path.modelName)
    modelName = config.path.modelName
    load = config.my_pruning.restart.load
    fil_it = os.path.join(snp, f"{modelName}_it{param.iterr}.txt")
    model = torch.load(param.load)  # Загрузка модели (модель осталась порезанной)
    # Кастыль для сегментации в офе
    if config.model.type_save_load == "interface":
        model.backbone_hooks._attach_hooks()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(f"{load.split('.')[0]}.msk", "r") as fp:
        sours_mask = json.load(fp)

    if do[0] * do[1] == param.orig_size:
        compres3(model, masks, sours_mask)  # Запуск прунинга

    model.to(device)  # Замененные слои не на карте
    # Узнать размер сверток после прунинга
    for name in get_stract(model):
        if name[0].split(".weight")[0] == config_list[0]["op_names"][0]:
            posle = name[2]
            break

    # После обрезки сеть обрезслась и кратна alf
    if do != posle and posle[1] % alf == 0:
        # Дообучаем
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model, loss, acc, st, time_elapsed2 = trainer.retrainer(
            model,
            optimizer,
            trainer.criterion,
            num_epochs=num_epochs,
            ind=param.self_ind,
        )
        # Запись результата
        f = open(fil_it, "a")
        strok = f"{str(param.self_ind)} {config_list[0]['op_names'][0]} {acc} {do} {posle}\n"
        f.write(strok)
        f.close()
        # Сохранение модели
        # Кастыль для сегментации в офе
        if config.model.type_save_load == "interface":
            model.backbone_hooks._clear_hooks()
        torch.save(
            model,
            os.path.join(
                snp,
                f"{modelName}_{config_list[0]['op_names'][0]}_it_{param.iterr}_acc_{acc:.3f}.pth",
            ),
        )
    else:
        # Запись размерностей до и после прунинга
        f = open(fil_it, "a")
        strok = (
            f"{str(param.self_ind)} {config_list[0]['op_names'][0]} EROR {do} {posle}\n"
        )
        f.write(strok)
        f.close()


def main(param):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(param.cuda)
    import torch
    import nni
    from nni.algorithms.compression.v2.pytorch.pruning import TaylorFOWeightPruner
    from nni.algorithms.compression.v2.pytorch.pruning import L2NormPruner

    # from nni.algorithms.compression.pytorch.pruning import L2FilterPruner
    from nni.compression.pytorch import ModelSpeedup
    import training as trainer
    import yaml
    import sys
    import copy
    from constants import Config

    config = yaml.safe_load(open(param.config, encoding ='utf-8'))
    config = Config(**config)

    mask = config.mask.type
    sys.path.append(config.model.path_to_resurs)

    # Параметры прунинга nni
    config_list = [
        {
            "sparsity": param.sparsity,
            "op_types": ["Conv2d"],
            "op_names": [param.name_sloi],
        }
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(param.load)
    # Кастыль для сегментации в офе
    if config.model.type_save_load == "interface":
        model.backbone_hooks._attach_hooks()
    # Кастыль для сегментации в офе
    if config.model.type_save_load == "interface":
        model.backbone_hooks._attach_hooks()
    model = model.to(device)
    traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())

    # Запоминание размерности слоя до прунинга
    do = posle = 0
    for name in get_stract(model):
        if name[0].split(".weight")[0] == config_list[0]["op_names"][0]:
            do = name[2]
    # Выбор алгоритма прунинга
    pruner = None
    if param.algoritm == "TaylorFOWeight":
        m = copy.deepcopy(model)
        pruner = TaylorFOWeightPruner(
            model,
            config_list,
            trainer.retrainer,
            traced_optimizer,
            trainer.criterion,
            training_batches=trainer.batch_size_t,
        )
    elif param.algoritm == "L2Norm":
        pruner = L2NormPruner(model, config_list)

    # Запуск прунинга от nni
    model, masks = pruner.compress()
    pruner._unwrap_model()
    # print(torch.nonzero(torch.sum(masks[list(masks.keys())[0]]['weight'],dim = (1,2,3))).shape[0])
    # print(masks[list(masks.keys())[0]]['weight'].shape[0])
    # print(torch.nonzero(torch.sum(masks[list(masks.keys())[0]]['weight'],dim = (1,2,3))).shape[0] / masks[list(masks.keys())[0]]['weight'].shape[0])
    # print(torch.nonzero(torch.sum(masks[list(masks.keys())[0]]['weight'],dim = (1,2,3))).shape[0] / masks[list(masks.keys())[0]]['weight'].shape[0] // 0.0001)
    # print((1 - param.sparsity))
    # print((1 - param.sparsity)// 0.0001)
    # # Если TaylorFOWeight обрезал некоректно
    if (
        param.algoritm == "TaylorFOWeight"
        and torch.nonzero(
            torch.sum(masks[list(masks.keys())[0]]["weight"], dim=(1, 2, 3))
        ).shape[0]
        / masks[list(masks.keys())[0]]["weight"].shape[0]
        // 0.0001
        != (1 - param.sparsity) // 0.0001
    ):
        print("ERROR in TaylorFOWeight algorithm. Running the L2NormPruner algorithm.")
        pruner = L2NormPruner(m, config_list)
        model, masks = pruner.compress()
        pruner._unwrap_model()

    type_pruning = ""
    if mask == "None":
        # Обрезка сети на основе маски от nni
        try:
            # Идеальная обрезка (только связанных слоёв)
            pruning_type(
                model, masks, do, param, config_list, config, type_pruning="defolt"
            )
        except:
            try:
                # Обрезка по особенностям ofa
                pruning_type(
                    model, masks, do, param, config_list, config, type_pruning="ofa"
                )
            except:
                # Крайне плохая, но точно работающая обрезка
                pruning_type(
                    model, masks, do, param, config_list, config, type_pruning="total"
                )

    elif mask == "mask":
        pruning_mask(model, masks, do, param, config_list, config)


if __name__ == "__main__":
    param = get_param()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{param.cuda}"
    main(param)
