import dearpygui.dearpygui as dpg
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
import time
import yaml
from code.utils import helps, retasc
from code.constants import Config
from ast import literal_eval
import subprocess
from multiprocessing import Process
import os
import pandas as pd
import glob
from pathlib import Path

def potok(config):
    return subprocess.call(
        [
            "python",
            "code",
            "--config",
            str(config),
        ],
    )


def main():
    font_skale = 0.5
    len_input_int = 110
    len_input_flost = 200
    len_input_text = 500
    len_input_min_text = 40
    help_i = 0

    dpg.create_context()
    with dpg.font_registry():
        with dpg.font(f'segoeui.ttf', 50, default_font=True, tag="Default font") as f:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)

    dpg.bind_font("Default font")
    dpg.set_global_font_scale(font_skale)

    def _help(message):
        last_item = dpg.last_item()
        group = dpg.add_group(horizontal=True)
        dpg.move_item(last_item, parent=group)
        dpg.capture_next_item(lambda s: dpg.move_item(s, parent=group))
        t = dpg.add_text("(?)", color=[0, 255, 0])
        with dpg.tooltip(t):
            dpg.add_text(message)

    def _log_name(sender, app_data, user_data):
        exp_save = dpg.get_value('path.exp_save')
        model_name = dpg.get_value('path.modelName')
        defolt_model_path =  os.path.join(exp_save, model_name, 'orig_model.pth')
        dpg.set_value('my_pruning.restart.load',defolt_model_path)
    def select_direct_model(sender, app_data, user_data):  
        filename = askdirectory()
        dpg.set_value(user_data, filename)
        exp_save = dpg.get_value('path.exp_save')
        model_name = dpg.get_value('path.modelName')
        defolt_model_path = os.path.join(exp_save, model_name, 'orig_model.pth')
        dpg.set_value('my_pruning.restart.load',defolt_model_path)

    def _log(sender, app_data, user_data): pass
    def _radio(sender, app_data, user_data): pass

    def select(sender, app_data, user_data):  
        filename = askdirectory()
        dpg.set_value(user_data, filename)
    def select_and_fils(sender, app_data, user_data):
        fils, direct = user_data
        filename = askopenfilename()
        full_name = os.path.basename(filename)
        name = os.path.splitext(full_name)[0]
        dpg.set_value(direct, name)
        dpg.set_value(fils, Path(filename).parent)
    def select_and_fils_2(sender, app_data, user_data):
        fils, direct = user_data
        filename = askopenfilename()
        full_name = os.path.basename(filename)
        dpg.set_value(direct, full_name)
        dpg.set_value(fils, Path(filename).parent)

    def select_file(sender, app_data, user_data):  
        filename = askopenfilename()
        dpg.set_value(user_data, filename)
    def select_files(sender, app_data, user_data):  
        filename = askopenfilenames()
        dpg.set_value(user_data, filename)
    def start(sender, app_data, user_data):  
        config = Config()
        
        config.task.type = retasc[dpg.get_value('task.type')]

        config.path.exp_save = dpg.get_value('path.exp_save')
        config.path.modelName = dpg.get_value('path.modelName')
        config.class_name = literal_eval(dpg.get_value('class_name'))

        config.model.type_save_load = dpg.get_value('model.type_save_load')
        config.model.path_to_resurs = dpg.get_value('model.path_to_resurs')
        config.model.name_resurs = dpg.get_value('model.name_resurs')
        config.model.size = [int(dpg.get_value('model.size[0]')), int(dpg.get_value('model.size[1]'))]
        config.model.gpu = dpg.get_value('model.gpu')

        config.dataset.num_classes = dpg.get_value('dataset.num_classes')
        config.dataset.annotation_path = dpg.get_value('dataset.annotation_path')
        config.dataset.annotation_name = dpg.get_value('dataset.annotation_name')

        config.retraining.num_epochs = dpg.get_value('retraining.num_epochs')
        config.retraining.lr = dpg.get_value('retraining.lr')
        config.retraining.dataLoader.batch_size_t = dpg.get_value('retraining.dataLoader.batch_size_t')
        config.retraining.dataLoader.num_workers_t = dpg.get_value('retraining.dataLoader.num_workers_t')
        config.retraining.dataLoader.pin_memory_t = dpg.get_value('retraining.dataLoader.pin_memory_t')
        config.retraining.dataLoader.drop_last_t = dpg.get_value('retraining.dataLoader.drop_last_t')
        config.retraining.dataLoader.shuffle_t = dpg.get_value('retraining.dataLoader.shuffle_t')
        config.retraining.dataLoader.batch_size_v = dpg.get_value('retraining.dataLoader.batch_size_v')
        config.retraining.dataLoader.num_workers_v = dpg.get_value('retraining.dataLoader.num_workers_v')
        config.retraining.dataLoader.pin_memory_v = dpg.get_value('retraining.dataLoader.pin_memory_v')
        config.retraining.dataLoader.drop_last_v = dpg.get_value('retraining.dataLoader.drop_last_v')
        config.retraining.dataLoader.shuffle_v = dpg.get_value('retraining.dataLoader.shuffle_v')
        
        config.training.dataLoader.batch_size_t = dpg.get_value('training.dataLoader.batch_size_t')
        config.training.dataLoader.num_workers_t = dpg.get_value('training.dataLoader.num_workers_t')
        config.training.dataLoader.pin_memory_t = dpg.get_value('training.dataLoader.pin_memory_t')
        config.training.dataLoader.drop_last_t = dpg.get_value('training.dataLoader.drop_last_t')
        config.training.dataLoader.shuffle_t = dpg.get_value('training.dataLoader.shuffle_t')
        config.training.dataLoader.batch_size_v = dpg.get_value('training.dataLoader.batch_size_v')
        config.training.dataLoader.num_workers_v = dpg.get_value('training.dataLoader.num_workers_v')
        config.training.dataLoader.pin_memory_v = dpg.get_value('training.dataLoader.pin_memory_v')
        config.training.dataLoader.drop_last_v = dpg.get_value('training.dataLoader.drop_last_v')
        config.training.dataLoader.shuffle_v = dpg.get_value('training.dataLoader.shuffle_v')
        config.algorithm = 'My_pruning'
        config.my_pruning.alf = dpg.get_value('my_pruning.alf')
        config.my_pruning.P = dpg.get_value('my_pruning.P')
        config.my_pruning.cart = literal_eval(dpg.get_value('my_pruning.cart'))
        config.my_pruning.iskl = literal_eval(dpg.get_value('my_pruning.iskl'))
        config.my_pruning.algoritm = dpg.get_value('my_pruning.algoritm')
        config.my_pruning.resize_alf = dpg.get_value('my_pruning.resize_alf')
        config.my_pruning.delta_crop = dpg.get_value('my_pruning.delta_crop')
        config.my_pruning.restart.start_iteration = dpg.get_value('my_pruning.restart.start_iteration')
        config.my_pruning.restart.load = dpg.get_value('my_pruning.restart.load')
        with open('config.yaml', 'w') as file:
            yaml.dump(config.dict(), file)
        p = Process(target=potok, args=['config.yaml'])
        p.start()
        dpg.configure_item("start", show=False)
        dpg.set_value('tab_bar', 'shov_logs')

    def reset_log(sender, app_data, user_data):
        exp_save = dpg.get_value('head.path.exp_save')
        model_name = dpg.get_value('head.path.modelName')
        dpg.delete_item("statistica")
        with dpg.group(tag="statistica", parent =  'Mstatistica'):
            df = pd.read_csv(
                os.path.join(
                    exp_save,
                    f"{model_name}_log.csv"
                    )
                )
            arr = df.to_numpy()
            with dpg.table():
                for i in range(df.shape[1]):
                    dpg.add_table_column(label=df.columns[i])
                for i in range(df.shape[0]):
                    with dpg.table_row():
                        for j in range(df.shape[1]):
                            dpg.add_text(f"{arr[i,j]}") 
            with dpg.tree_node(label="Метрика от итерации"):
                with dpg.plot(label="Метрика от итерации", height=500, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Итерация")
                    with dpg.plot_axis(dpg.mvYAxis):
                        dpg.add_line_series(list(df['N'].values), list(df['acc'].values), label="Метрика")
                        dpg.add_line_series(list(df['N'].values), list(df['size'].values), label="Сжатие")
            with dpg.tree_node(label="Метрика от степини обрезки"):
                with dpg.plot(label="Метрика от степини обрезки", height=500, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Степень обрезки")
                    with dpg.plot_axis(dpg.mvYAxis, label="Метрика"):
                        dpg.add_line_series(list(1 - df['size'].values), list(df['acc'].values), label="Метрика от степини обрезки") 
            with dpg.tree_node(label=f"Данные по итерациям", default_open=False):
                fils = glob.glob(os.path.join(
                    exp_save,
                    model_name,
                    '*.csv')
                )
                for i in range(len(fils)):
                    with dpg.tree_node(label=f"Итерация {i}", default_open=False):
                        df = pd.read_csv(os.path.join(
                                                        exp_save,
                                                        model_name,
                                                        f'{model_name}_it{i}.csv'))
                        arr = df.to_numpy()
                        with dpg.table():
                            for i in range(df.shape[1]):
                                dpg.add_table_column(label=df.columns[i])
                            for i in range(df.shape[0]):
                                with dpg.table_row():
                                    for j in range(df.shape[1]):
                                        dpg.add_text(f"{arr[i,j]}") 
                        with dpg.tree_node(label=f"График", default_open=False):
                            with dpg.plot(height=500, width=-1):

                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Индекс обрезки", no_gridlines=True)
                                dpg.set_axis_limits(dpg.last_item(), 0, df.shape[0] -1)
                                
                                t = []
                                for i in range(df['ind'].shape[0]):
                                    t.append((str(df['ind'].values[i]), i))
                                dpg.set_axis_ticks(dpg.last_item(),tuple(t))
                                # create y axis
                                with dpg.plot_axis(dpg.mvYAxis, label="Метрика"):
                                    dpg.set_axis_limits(dpg.last_item(), 0, 1)
                                    for i in range(df['name'].shape[0]):
                                        dpg.add_bar_series([i], [df['acc'].values[i]], weight=1)

    def graf(sender, app_data, user_data):
        from torch import load
        from code.my_pruning_pabotnik import symbolic_trace
        exp_save = dpg.get_value('debag.path.exp_save')
        class_name = literal_eval(dpg.get_value('debag.class_name'))
        model = load(exp_save)
        gm = symbolic_trace(model, concrete_args= None, class_name = class_name)
        node_specs = [[n.op, n.name, str(n.target), n.args, n.kwargs]
                    for n in gm.graph.nodes ]
        dpg.delete_item("debag")
        with dpg.group(tag="debag", parent =  'Mdebag'):
            with dpg.table():
                for i in ['opcode', 'name', 'target', 'args', 'kwargs']:
                    dpg.add_table_column(label=i)
                for i in range(len(node_specs)):
                    with dpg.table_row():
                        for j in range(len(node_specs[0])):
                            dpg.add_text(f"{node_specs[i][j]}")

    def mask(sender, app_data, user_data):
        from torch import load
        from code.my_pruning_pabotnik import get_mask
        exp_save = dpg.get_value('debag.path.exp_save')
        class_name = literal_eval(dpg.get_value('debag.class_name'))
        model = load(exp_save)
        sours_mask = get_mask(model, class_name = class_name)
        dpg.delete_item("debag")
        with dpg.group(tag="debag", parent =  'Mdebag'):
            for i in sours_mask:
                with dpg.tree_node(label=i[0][0]):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            for j in i:
                                dpg.add_text(j[0])
                        with dpg.group():
                            for j in i:
                                dpg.add_text(j[1])
    
    def not_prun(sender, app_data, user_data):
        from torch import load, rand
        from code.my_pruning_pabotnik import delet, get_mask
        import copy
        device = "cpu"
        exp_save = dpg.get_value('debag.path.exp_save')
        class_name = literal_eval(dpg.get_value('debag.class_name'))
        model = load(exp_save)
        sours_mask = get_mask(model, class_name = class_name)
        shape = literal_eval(dpg.get_value('debag.shape'))

        dpg.delete_item("debag")
        with dpg.group(tag="debag", parent =  'Mdebag'):
            inp = rand(*shape).to(device)
            del_ind = [0,1,2]
            for k, crop in enumerate(sours_mask[:-1]):
                model_crop = copy.deepcopy(model)
                try:
                    for sloi, i in crop:
                        # print(sloi, crop)
                        if i == 0:
                            delet(model_crop, sloi, i = del_ind)
                        elif i == 1:
                            delet(model_crop, sloi, j = del_ind)
                    model_crop = model_crop.to(device)
                    out = model_crop(inp)
                except:
                     with dpg.tree_node(label=sours_mask[k][0]):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                for j in sours_mask[k]:
                                    dpg.add_text(j[0])
                            with dpg.group():
                                for j in sours_mask[k]:
                                    dpg.add_text(j[1])
            dpg.add_text('End')
    def print_net():
        from torch import load
        exp_save = dpg.get_value('debag.path.exp_save')
        model = load(exp_save)
        dpg.delete_item("debag")
        with dpg.group(tag="debag", parent =  'Mdebag'):
            dpg.add_text(str(model.__repr__))
    def get_flops():
        from torch import load
        from code.my_pruning_pabotnik import get_size
        exp_save = dpg.get_value('debag.path.exp_save')
        class_name = literal_eval(dpg.get_value('debag.class_name'))
        model = load(exp_save)
        shape = literal_eval(dpg.get_value('debag.shape'))
        dpg.delete_item("debag")
        
        with dpg.group(tag="debag", parent =  'Mdebag'):
            dpg.add_input_text(default_value=f"{get_size(model, shape[2:])}")
    def stract():
        from torch import load
        from code.my_pruning_pabotnik import get_stract
        exp_save = dpg.get_value('debag.path.exp_save')
        class_name = literal_eval(dpg.get_value('debag.class_name'))
        model = load(exp_save)
        shape = literal_eval(dpg.get_value('debag.shape'))
        dpg.delete_item("debag")
        stract = get_stract(model)
        with dpg.group(tag="debag", parent =  'Mdebag'):
            with dpg.table():
                for i in ['имя', 'тип', 'размерность', 'ядро']:
                    dpg.add_table_column(label=i)
                for i in range(len(stract)):
                    with dpg.table_row():
                        for j in range(len(stract[i])):
                            dpg.add_text(f"{stract[i][j]}")

    def onnx():
        import torch
        inport = dpg.get_value('onnx.path.inport')
        inport = literal_eval('['+inport[1:-1]+']')
        export = dpg.get_value('onnx.path.export')
        shape = literal_eval(dpg.get_value('onnx.shape'))
        dummy_input = torch.randn(*shape, dtype=torch.float).to("cpu")
        input_names = ["input"]
        output_names = ["output"]
        dpg.delete_item("onnx")
        with dpg.group(tag="onnx", parent =  'Monnx'):
            for mame in inport:
                model = torch.load(mame, map_location="cpu")
                save = os.path.join(export, os.path.basename(mame).split('.pth')[0]+".onnx")
                torch.onnx.export(model, dummy_input,
                                save,
                                verbose=True, input_names=input_names, output_names=output_names)
                dpg.add_text(save)
            dpg.add_text('Готово')
    
    with dpg.window(tag="main", menubar=True):
        
        with dpg.menu_bar():
            with dpg.menu(label="Tools"):
                dpg.add_menu_item(label="Показать меню стилей", callback=lambda:dpg.show_tool(dpg.mvTool_Style))
                dpg.add_menu_item(label="Показать меню шрифтов", callback=lambda:dpg.show_tool(dpg.mvTool_Font))
            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Обработка во время импута", check=True, callback=lambda s, a: dpg.configure_app(wait_for_input=a))
                dpg.add_menu_item(label="На весь экран", callback=lambda:dpg.toggle_viewport_fullscreen())
        with dpg.tab_bar(tag='tab_bar'):
            with dpg.tab(label="Прунинг"):
                with dpg.tree_node(label="Основные параметры", default_open=True,tag='tree_node_1'):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text('Задача компьютерного зрения')
                            dpg.add_text('Название эксперимента')
                            dpg.add_text('Путь к сохраняемым данным')
                            dpg.add_text('Добавленные классы параметров')
                            dpg.add_text('Видеокарта основного процесса')
                        with dpg.group():
                            dpg.add_radio_button(("Классификация", "Сегментация", "Детекция"), callback=_radio, horizontal=True, tag="task.type", default_value="Классификация")
                            _help(helps[help_i]); help_i +=1
                            dpg.add_input_text(default_value ="model", callback=_log_name, width = len_input_text, no_spaces=True, tag="path.modelName")
                            
                            with dpg.group(horizontal=True):
                                dpg.add_input_text(default_value = os.path.join('snp', time.strftime("%d_%m_%Y_%H_%M", time.localtime(time.time()))), width = len_input_text, callback=_log_name, no_spaces=True, tag = 'path.exp_save')
                                dpg.add_button(label="Найти", callback=select_direct_model, user_data='path.exp_save')
                            dpg.add_input_text(default_value = "['timm.layers.norm_act']", width = len_input_text, callback=_log, no_spaces=True, tag = 'class_name')
                            _help(helps[help_i]); help_i +=1
                            dpg.add_input_int(callback=_log, width = len_input_int, default_value = 0, min_value = 0, min_clamped = True, tag = 'model.gpu')
                with dpg.tree_node(label="Модель", default_open=True,tag='tree_node_2'):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text('Тип модели')
                            dpg.add_text('Название модели')
                            dpg.add_text('Путь к мадели')
                            dpg.add_text('Размер входа')
                        with dpg.group():
                            dpg.add_radio_button(("pth", "interface"), callback=_radio, horizontal=True, tag="model.type_save_load", default_value="pth")
                            _help(helps[help_i]); help_i +=1
                            dpg.add_input_text(default_value ="timm_resnet18", callback=_log, width = len_input_text, no_spaces=True, tag="model.name_resurs")
                            _help(helps[help_i]); help_i +=1
                            with dpg.group(horizontal=True):
                                dpg.add_input_text(default_value = "models", width = len_input_text, callback=_log, no_spaces=True, tag = 'model.path_to_resurs')
                                _help(helps[help_i]); help_i +=1
                                dpg.add_button(label="Найти", callback=select_and_fils, user_data=['model.path_to_resurs', 'model.name_resurs'])
                            with dpg.group(horizontal=True):
                                dpg.add_input_text(callback=_log, default_value="224", scientific=True, width = len_input_min_text, tag = 'model.size[0]')
                                dpg.add_text(':')
                                dpg.add_input_text(callback=_log, default_value="224", scientific=True, width = len_input_min_text, tag = 'model.size[1]')
                                _help(helps[help_i]); help_i +=1
                with dpg.tree_node(label="Набор данных", default_open=True,tag='tree_node_3'):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text('Количество классов')
                            dpg.add_text('Название csv файла')
                            dpg.add_text('Путь к данным')
                        with dpg.group():
                            dpg.add_input_int(callback=_log, width = len_input_int, default_value = 10, min_value = 1, min_clamped = True, tag = 'dataset.num_classes')
                            dpg.add_input_text(default_value ="data.csv", callback=_log,width = len_input_text, no_spaces=True, tag = 'dataset.annotation_name')
                            with dpg.group(horizontal=True):
                                with dpg.group(tag = 'Data'):
                                    dpg.add_input_text(default_value = os.path.join('D:/', 'db', 'ImageNette_10'), width = len_input_text, callback=_log, no_spaces=True, tag = 'dataset.annotation_path')
                                dpg.add_button(label="Найти", callback=select_and_fils_2, user_data=['dataset.annotation_path','dataset.annotation_name'])
                with dpg.tree_node(label="Конфигурация дообучения", default_open=True,tag='tree_node_4'):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text('Количество эпох')
                            dpg.add_text('learning rate')
                        with dpg.group():
                            dpg.add_input_int(callback=_log, width = len_input_int, default_value = 1, min_value = 1, min_clamped = True, tag = 'training.num_epochs')
                            dpg.add_input_float(callback=_log, width = len_input_flost, default_value = 0.00001, format="%.8f", min_value = 0, min_clamped = True, tag = 'training.lr')
                    with dpg.tree_node(label="тренеровка", default_open=False):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text('Размер батча')
                                dpg.add_text('num_workers')
                                dpg.add_text('pin_memory')
                                dpg.add_text('drop_last')
                                dpg.add_text('shuffle')
                            with dpg.group():
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 10, min_value = 1, min_clamped = True, tag = 'training.dataLoader.batch_size_t')
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 1, min_value = 0, min_clamped = True, tag = 'training.dataLoader.num_workers_t')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'training.dataLoader.pin_memory_t')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'training.dataLoader.drop_last_t')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'training.dataLoader.shuffle_t')
                    with dpg.tree_node(label="валидация", default_open=False):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text('Размер батча')
                                dpg.add_text('num_workers')
                                dpg.add_text('pin_memory')
                                dpg.add_text('drop_last')
                                dpg.add_text('shuffle')
                            with dpg.group():
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 10, min_value = 1, min_clamped = True, tag = 'training.dataLoader.batch_size_v')
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 1, min_value = 0, min_clamped = True, tag = 'training.dataLoader.num_workers_v')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'training.dataLoader.pin_memory_v')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'training.dataLoader.drop_last_v')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'training.dataLoader.shuffle_v')
                with dpg.tree_node(label="Конфигурация востановления", default_open=False,tag='tree_node_5'):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text('Количество эпох')
                            dpg.add_text('learning rate')
                        with dpg.group():
                            dpg.add_input_int(callback=_log, width = len_input_int, default_value = 1, min_value = 1, min_clamped = True, tag = 'retraining.num_epochs')
                            dpg.add_input_float(callback=_log, width = len_input_flost, default_value = 0.00001, format="%.8f", tag = 'retraining.lr')
                    with dpg.tree_node(label="тренеровка", default_open=False):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text('Размер батча')
                                dpg.add_text('num_workers')
                                dpg.add_text('pin_memory')
                                dpg.add_text('drop_last')
                                dpg.add_text('shuffle')
                            with dpg.group():
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 10, min_value = 1, min_clamped = True, tag = 'retraining.dataLoader.batch_size_t')
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 1, min_value = 0, min_clamped = True, tag = 'retraining.dataLoader.num_workers_t')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'retraining.dataLoader.pin_memory_t')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'retraining.dataLoader.drop_last_t')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'retraining.dataLoader.shuffle_t')
                    with dpg.tree_node(label="валидация", default_open=False):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text('Размер батча')
                                dpg.add_text('num_workers')
                                dpg.add_text('pin_memory')
                                dpg.add_text('drop_last')
                                dpg.add_text('shuffle')
                            with dpg.group():
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 10, min_value = 1, min_clamped = True, tag = 'retraining.dataLoader.batch_size_v')
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 1, min_value = 0, min_clamped = True, tag = 'retraining.dataLoader.num_workers_v')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'retraining.dataLoader.pin_memory_v')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'retraining.dataLoader.drop_last_v')
                                dpg.add_checkbox( callback=_log, default_value=True, tag = 'retraining.dataLoader.shuffle_v')
                with dpg.tree_node(label="Параметры алгоритма", default_open=False,tag='tree_node_6'):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text('Кратность каналов')
                            dpg.add_text('Коэфициент сжатия')
                            dpg.add_text('Карты')
                            dpg.add_text('Исключения')
                            dpg.add_text('Алгоритм')
                            dpg.add_text('resize_alf')
                            dpg.add_text('delta_crop')
                        with dpg.group():
                            dpg.add_input_int(callback=_log, width = len_input_int, default_value = 32, min_value = 1, min_clamped = True, tag = 'my_pruning.alf')
                            _help(helps[help_i]); help_i +=1
                            dpg.add_input_float(callback=_log, width = len_input_flost, default_value = 0.99, min_value = 0, min_clamped = True, max_value=1,max_clamped=True, tag = 'my_pruning.P')
                            _help(helps[help_i]); help_i +=1
                            dpg.add_input_text(default_value = '[0,0,0]', width = len_input_int, callback=_log, no_spaces=True, tag = 'my_pruning.cart')
                            _help(helps[help_i]); help_i +=1
                            dpg.add_input_text(default_value = '[]', width = len_input_int, callback=_log, no_spaces=True, tag = 'my_pruning.iskl')
                            _help(helps[help_i]); help_i +=1
                            dpg.add_radio_button(("TaylorFOWeight", "L2Norm"), callback=_radio, horizontal=True, tag="my_pruning.algoritm", default_value="TaylorFOWeight")
                            _help(helps[help_i]); help_i +=1
                            dpg.add_checkbox( callback=_log, tag="my_pruning.resize_alf")
                            _help(helps[help_i]); help_i +=1
                            dpg.add_input_float(callback=_log, width = len_input_flost, default_value = 0.1, min_value = 0.001, min_clamped = True, max_value=1,max_clamped=True, tag="my_pruning.delta_crop")
                            _help(helps[help_i]); help_i +=1
                    with dpg.tree_node(label="Рестарт", default_open=False):
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text('Номер итерации')
                                dpg.add_text('Модель с которой продолжить обучение')
                            with dpg.group():
                                dpg.add_input_int(callback=_log, width = len_input_int, default_value = 0, min_value = 0, min_clamped = True, tag = 'my_pruning.restart.start_iteration')
                                with dpg.group(horizontal=True):
                                    dpg.add_input_text(default_value = os.path.join('snp', time.strftime("%d_%m_%Y_%H_%M", time.localtime(time.time())),'model', 'orig_model.pth'), width = len_input_text, callback=_log, no_spaces=True, tag = 'my_pruning.restart.load')
                                    dpg.add_button(label="Найти", callback=select_file, user_data='my_pruning.restart.load')
                with dpg.table(tag='Goods', header_row=False, borders_innerH=False, borders_outerH=False, borders_innerV=False, borders_outerV=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_table_cell()
                        dpg.add_button(label="Старт", callback=start, width=300, height= 100, tag = 'start')
            with dpg.tab(label="Просмотр", tag="shov_logs"):
                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text('Название эксперимента')
                            dpg.add_text('Путь к сохраняемым данным')
                        with dpg.group():
                            dpg.add_input_text(default_value ="model", callback=_log_name, width = len_input_text, no_spaces=True, tag="head.path.modelName")
                            with dpg.group(horizontal=True):
                                dpg.add_input_text(default_value = os.path.join('snp', time.strftime("%d_%m_%Y_%H_%M", time.localtime(time.time()))), width = len_input_text, callback=_log_name, no_spaces=True, tag = 'head.path.exp_save')
                                dpg.add_button(label="Найти", callback=select_direct_model, user_data='head.path.exp_save')
                dpg.add_button(label="Обновить", callback=reset_log)
                with dpg.group(tag="Mstatistica"):
                    pass
                with dpg.group(tag="statistica"):
                    pass
            with dpg.tab(label="Дебаг"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text('Путь к модели')
                        dpg.add_text('Добавленные классы параметров')
                        dpg.add_text('Размер входа')
                    with dpg.group():
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(width = len_input_text, callback=_log, no_spaces=True, tag = 'debag.path.exp_save')
                            dpg.add_button(label="Найти", callback=select_file, user_data='debag.path.exp_save')
                        dpg.add_input_text(default_value = "['timm.layers.norm_act']", width = len_input_text, callback=_log, no_spaces=True, tag = 'debag.class_name')
                        _help(helps[1])
                        dpg.add_input_text(default_value = "[1,3,224,224]", width = len_input_text, callback=_log, no_spaces=True, tag = 'debag.shape')
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Построить граф", callback=graf)
                    dpg.add_button(label="Получить маску", callback=mask)
                    dpg.add_button(label="Найти непрунящиеся блоки", callback=not_prun)
                    dpg.add_button(label="Распечатать сеть", callback=print_net)
                    dpg.add_button(label="Посчитать flops", callback=get_flops)
                    dpg.add_button(label="Построисть структуру", callback=stract)
                with dpg.group(tag="Mdebag"):
                    pass
                with dpg.group(tag="debag"):
                    pass
            with dpg.tab(label="Конвертация в ONNX"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text('Пути к моделям')
                        dpg.add_text('Путь экспорта')
                        dpg.add_text('Размер входа')
                    with dpg.group():
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(width = len_input_text, callback=_log, no_spaces=True, tag = 'onnx.path.inport')
                            dpg.add_button(label="Найти", callback=select_files, user_data='onnx.path.inport')
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(width = len_input_text, callback=_log, no_spaces=True, tag = 'onnx.path.export')
                            dpg.add_button(label="Найти", callback=select, user_data='onnx.path.export')
                        dpg.add_input_text(default_value = "[1,3,224,224]", width = len_input_text, callback=_log, no_spaces=True, tag = 'onnx.shape')
                dpg.add_button(label="Старт", callback=onnx)
                with dpg.group(tag="Monnx"):
                    pass
                with dpg.group(tag="onnx"):
                    pass
            with dpg.tab(label="Обучение"):
                pass
    # dpg.show_item_registry()
    dpg.create_viewport(title='U-Pruner')
    dpg.setup_dearpygui()
    dpg.set_primary_window('main', True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()