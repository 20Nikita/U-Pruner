# -*- coding: utf-8 -*-
import time
import copy
import torch
import numpy as np

def train_model(model, 
                classification_criterion, 
                optimizer, 
                train_dataloader, 
                val_dataloader, 
                batch_size_t, 
                batch_size_v, 
                num_epochs=100,
                N_class = 2,
                rezim = ['T', 'V'],
                fil = None):
    # Запомнить время начала обучения
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mass = [[],[],[]]
    # Копировать параметры поданной модели
    best_model_wts_classification = copy.deepcopy(model.state_dict())
    best_Loss_classification = 10000.0         # Лучший покозатель модели
    best_epoch_classification = 0
    best_acc = 0         # Лучший покозатель модели
    best_epoch_classification = 0
    pihati = ""
    tit = "\n!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    
    for epoch in range(num_epochs):
        pihati += 'Epoch {}/{}'.format(epoch + 1, num_epochs) + "\n"
        pihati += '-' * 10 + "\n"
        f = open(fil, "w")
        f.write(pihati + "\n" + tit)
        f.close()
        
        # У каждой эпохи есть этап обучения и проверки
        for phase in rezim:
            if phase == 'T':
                dataloader = train_dataloader
                batch_size = batch_size_t
                dataset_sizes = len(train_dataloader) * batch_size
                model.train()  # Установить модель в режим обучения
            elif phase == 'V':
                dataloader = val_dataloader
                batch_size = batch_size_v
                dataset_sizes = len(val_dataloader) * batch_size
                model.eval()   #Установить модель в режим оценки
            
            # Обнуление параметров
            running_classification_loss = 0.0
            running_corrects = 0
            iiter = 0
            # Получать порции картинок и иx классов из датасета
            for inputs, classification_label in dataloader:
                iiter += 1
                f = open(fil, "w")
                f.write('Epoch {}/{}'.format(epoch + 1, num_epochs) + "\n" 
                        + str(iiter)+"/"+str(int(len(dataloader))) + "\n" 
                        + "\n" 
                        + pihati + "\n" 
                        + tit)
                f.close()
                
                classification_label = classification_label.to(device)
                    
                    
                # считать все на видеокарте или ЦП
                inputs = inputs.to(device)
                classification_label = classification_label.to(device)
                # обнулить градиенты параметра
                optimizer.zero_grad()
                # forward
                # Пока градиент можно пощитать, шитать только на учимся
                with torch.set_grad_enabled(phase == 'T'):
                    # Проход картинок через модель
                    classification = model(inputs)
                    # Получить индексы максимальных элементов
                    _, preds = torch.max(classification, 1)
                    loss = classification_criterion(classification, classification_label)
                    # Если учимся
                    if phase == 'T':
                        # Вычислить градиенты
                        loss.backward()
                        # Обновить веса
                        optimizer.step()
                # Статистика
                running_classification_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == classification_label.data)# Колличество правильных ответов
            # Усреднить статистику
            epoch_classification_loss = running_classification_loss / dataset_sizes
            running_classification_loss/= dataset_sizes
            epoch_acc = running_corrects / dataset_sizes
            
            pihati += '{}_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_classification_loss, epoch_acc)
            pihati += "\n"
            f = open(fil, "w")
            f.write(pihati + "\n" +  tit)
            f.close()
            
#             print('{}_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_classification_loss, epoch_acc))
            if phase == 'V':
                mass[0].append(epoch)
                mass[1].append(epoch_acc.item())
                mass[2].append(epoch_classification_loss)

            # Копироование весов успешной модели на вэйле
            if (phase == 'V') and epoch_classification_loss < best_Loss_classification:
                best_Loss_classification = epoch_classification_loss
                best_model_wts_classification = copy.deepcopy(model.state_dict())
                best_epoch_classification = epoch + 1
                
            # Копироование весов успешной модели на вэйле
            if (phase == 'V') and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts_acc = copy.deepcopy(model.state_dict())
                best_epoch_acc = epoch + 1
                
    # Конечное время и печать времени работы
    time_elapsed = time.time() - since
    overfit_model = model
    model1 = model
    model2 = model
    model2.load_state_dict(best_model_wts_classification)
    model1.load_state_dict(best_model_wts_acc)
    return model1, model2, overfit_model, pihati, mass, time_elapsed
