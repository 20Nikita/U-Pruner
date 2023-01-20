import argparse
import torch
import os
from ptflops import get_model_complexity_info
#Ïîëó÷èòü äàííûå îá îáó÷àåìûõ ïàðàìåòðàõ ñåòè
def get_stract(model):
    stact = []
    for name, parametr in model.named_parameters():
        struct = name.split('.')
        noda = model
        for bloc in struct:
            try:
                intt = int(bloc)
                type_int = True
            except:
                type_int = False

            if bloc != "weight" and bloc != "bias" and not type_int:
                noda = eval("noda.{}".format(bloc))
            elif type_int:
                noda = eval("noda[{}]".format(intt))
        types = str(type(noda)).split("'")[1]
        if types == 'torch.nn.modules.conv.Conv2d':
            stact.append([name,types,[noda.in_channels,noda.out_channels], noda.kernel_size[0] * noda.kernel_size[1]])
        elif types == 'torch.nn.modules.batchnorm.BatchNorm2d':
            stact.append([name,types,[noda.num_features]])
        elif types == 'torch.nn.modules.linear.Linear':
            stact.append([name,types,[noda.in_features,noda.out_features]])
        else:
            stact.append([name,types,"******************"])
#         for p in noda.named_parameters():
#             stact[-1].append([p[0],len(p[1])])
    return(stact)

# Ïîëó÷èòü ñóìàðíîå êîëè÷åñòâî ñòðîê â óîíâàëþöèîííîé ìàòðèöå. Ìåðà ðàçìåðà ñåòè
def get_size(model):
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False, verbose=False)
    return params
    #N = 0
    #for name in get_stract(model):
    #    if name[1]=='torch.nn.modules.conv.Conv2d':
    #        N = N + name[2][1] * name[2][0] * name[3]
    #return N

# Óäàëèòü âåñà. Âõîä: weight - torch.nn.Parameter (Âåñà), 
#i - list (èíåêñû ñòðîê ìàññèâà âåñîâ, êîòîðûå íóæíî óäàëèòü), 
#j - list (èíåêñû ñòîëáöîâ ìàññèâà âåñîâ, êîòîðûå íóæíî óäàëèòü)
def delet_weight(weight, i = -1, j = -1):
    # Óäàëèòü ñòðîêè
    if i != -1:
        for k in range(len(i)):
            k = i[len(i) - 1 - k]
            weight = torch.nn.Parameter(torch.cat([weight[0:k],weight[k+1:]]))
    # Óäàëèòü ñòîëáöû
    if j != -1:
        # Çàìåíà èíäåêñîâ êîòîðûå íóæíî óäàëèòü íà èíäåêñû, êîòîðûå íóæíî îñòàâèòü
        j = set(range(len(weight[0])))-set(j)
        #Çàïîìíèòü ðàçìåðíîñòü
        weight_shape = list(weight.shape)
        weight_shape[1] = 1 # Óäàëÿåì ñòîëáöû
        #Ïåðåñîáðàòü âåñà èç èíäåêñîâ, êîòîðûå íóæíî îñòàâèòü
        weight = [weight[:,k,:,:].reshape(weight_shape) for k in j]
        weight = torch.nn.Parameter(torch.cat(weight,1))
    return weight

def delet_weight_Linear(weight, i = -1, j = -1):
    # Óäàëèòü ñòðîêè
    if i != -1:
        for k in range(len(i)):
            k = i[len(i) - 1 - k]
            weight = torch.nn.Parameter(torch.cat([weight[0:k],weight[k+1:]]))
    # Óäàëèòü ñòîëáöû
    if j != -1:
        # Çàìåíà èíäåêñîâ êîòîðûå íóæíî óäàëèòü íà èíäåêñû, êîòîðûå íóæíî îñòàâèòü
        j = set(range(len(weight[0])))-set(j)
        #Çàïîìíèòü ðàçìåðíîñòü
        weight_shape = list(weight.shape)
        weight_shape[1] = 1 # Óäàëÿåì ñòîëáöû
        #Ïåðåñîáðàòü âåñà èç èíäåêñîâ, êîòîðûå íóæíî îñòàâèòü
        weight = [weight[:,k].reshape(weight_shape) for k in j]
        weight = torch.nn.Parameter(torch.cat(weight,1))
    return weight

#Çàìåíà ñëîÿ òàêèìæå, íî ñ îáðåçàííûìè ïàðàìåòðàìè è óìåíüøåííûì ðàçìåðîì.    
def delet(model, Delet_Name_sloi, i = -1, j = -1):
    #Óçíàòü òèï ñëîÿ
    types = str(type(eval("model.{}".format(Delet_Name_sloi)))).split("'")[1]
    if types == 'torch.nn.modules.conv.Conv2d':
        groups = eval("model.{}".format(Delet_Name_sloi)).groups
        groups = 1 if groups == 1 else groups-len(i)
        # Åñëè â êâàäðàòíîé ìàòðèöå ñ 1 ïðîäóáëèðîâàííûì n ðàç ñòîëáöîì ìíå íóæíî óäàëèòü ñòîëáåö. ß óäàëþ ñòðîêó ïî èíäåêñó ñòîëáöà.
        #Ïðîáëåìà â òîì, ÷òî å¸ îòîáðàæàåìàÿ ðàçìåðíîñòü (a,b,c,d), à ôàêòè÷åñêèå âåñà (a,1,c,d) ñ groups = b
        if groups != 1 and i == -1:
            i = j
            j = -1
        
        # Èçìåíèòü ðàçìåð ñòîëáöîâ
        in_channels = eval("model.{}".format(Delet_Name_sloi)).in_channels
        in_channels = in_channels if j == -1 else in_channels-len(j)
        # Åñëè groups != 1, òî ýòî êâàäðàòíàÿ ìàòðèöà ñ ïðîäóáëèðîâàííûìè ñòîëáöàìè, à â íåé ÿ óäàëÿþ ñòðîêè ïî èíäåêñó ñòîëáöà
        in_channels = in_channels if groups == 1 else in_channels-len(i)

        # Èçìåíèòü ðàçìåð ñòðîê
        out_channels = eval("model.{}".format(Delet_Name_sloi)).out_channels
        out_channels = out_channels if i == -1 else out_channels-len(i)
        
        # Ýòè ïàðàìåòðû êîïèðóþ áåç èçìåíåíèé
        kernel_size = eval("model.{}".format(Delet_Name_sloi)).kernel_size
        stride = eval("model.{}".format(Delet_Name_sloi)).stride
        padding = eval("model.{}".format(Delet_Name_sloi)).padding
        dilation = eval("model.{}".format(Delet_Name_sloi)).dilation
        padding_mode = eval("model.{}".format(Delet_Name_sloi)).padding_mode
        
        # Ïîëó÷èòü îáðåçàííûå âåñà
        weight = delet_weight(eval("model.{}".format(Delet_Name_sloi)).weight, i, j)
        # bias - îáó÷àåìûé ïàðàìåòð, ðàçìåðíîñòüþ 1, ñ êîëè÷åñòâîì ýëåìåíòîâ = êîëè÷åñòâó ñòðîê â êîíâàëþöèîííîì ñëîå. Íî åãî ìîæåò íå áûòü.
        bias = eval("model.{}".format(Delet_Name_sloi)).bias
        bias_w = bias
        if bias != None:
            bias = True
            if i != -1:
                bias_w = delet_weight(eval("model.{}".format(Delet_Name_sloi)).bias, i, j)
        
        # Ñîçäàòü óêîïèþ ñëîÿ ñ îáðåçàííûìè ïàðàìåòðàìè, íî ðàíäîìíûìè âåñàìè
        new_pam = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, 
                                  stride = stride, padding = padding, dilation = dilation, groups = groups, 
                                  bias = bias, padding_mode = padding_mode)
        # Ïåðåäàòü âåñà
        new_pam.weight = weight
        new_pam.bias = bias_w
        #print(new_pam,new_pam.weight.shape)

        # Çàìåíèòü ñëîé îáðåçàííûì
        exec("model.{} = new_pam".format(Delet_Name_sloi))
        
    elif types == 'torch.nn.modules.batchnorm.BatchNorm2d':
        # Èçìåíèòü ðàçìåð ñòðîê
        num_features = eval("model.{}".format(Delet_Name_sloi)).num_features -len(i)
        
        # Ýòè ïàðàìåòðû êîïèðóþ áåç èçìåíåíèé
        eps = eval("model.{}".format(Delet_Name_sloi)).eps
        momentum = eval("model.{}".format(Delet_Name_sloi)).momentum
        affine = eval("model.{}".format(Delet_Name_sloi)).affine
        track_running_stats = eval("model.{}".format(Delet_Name_sloi)).track_running_stats
        
        # Ñîçäàòü óêîïèþ ñëîÿ ñ îáðåçàííûìè ïàðàìåòðàìè, íî ðàíäîìíûìè âåñàìè
        new_pam = torch.nn.BatchNorm2d(num_features, eps = eps, momentum = momentum, affine = affine, 
                                       track_running_stats = track_running_stats)
        # Ïåðåäàòü âåñà
        new_pam.weight = delet_weight(eval("model.{}".format(Delet_Name_sloi)).weight, i, j)
        new_pam.bias = delet_weight(eval("model.{}".format(Delet_Name_sloi)).bias, i, j)
        
        # Çàìåíèòü ñëîé îáðåçàííûì
        exec("model.{} = new_pam".format(Delet_Name_sloi))

    elif types == 'torch.nn.modules.linear.Linear':

        in_features = eval("model.{}".format(Delet_Name_sloi)).in_features
        in_features = in_features if i == -1 else in_features-len(i)

        out_features = eval("model.{}".format(Delet_Name_sloi)).out_features
        out_features = out_features if j == -1 else out_features-len(j)

        bias = eval("model.{}".format(Delet_Name_sloi)).bias

        new_pam = torch.nn.Linear(in_features, out_features)

        # Ïåðåäàòü âåñà
        new_pam.weight = delet_weight_Linear(eval("model.{}".format(Delet_Name_sloi)).weight, j, i)
        new_pam.bias = delet_weight_Linear(eval("model.{}".format(Delet_Name_sloi)).bias, j)

        # Çàìåíèòü ñëîé îáðåçàííûì
        exec("model.{} = new_pam".format(Delet_Name_sloi))

        

# Çàìåíèòü ñòðîêó ôîðìàòà *.n.*.m.* íà *[n].*[m].*
def rename(name):
    new_name = ""
    name_split = name.split(".")
    for i in range(len(name_split)):
        try:
            new_name = new_name + "[{}]".format(int(name_split[i]))
        except:
            new_name = new_name + ".{}".format(name_split[i])
    try:
        t = int(name_split[0])
        return new_name
    except:
        return new_name[1:]
    
# Èíâåíòèðîâàòü ñïèñîê ([a,b,c] => [c,b,a])
def obr(names):
    names_invert = names.copy()
    for i in range(len(names)):
        names_invert[len(names)-i-1] = names[i]
    return names_invert

# Êîíâåðòàöèÿ ìàñêè nni â èíäåêñû óäîëÿåìûõ ñòðîê
def get_delet(masks):
    delet = []
    for i in range(len(masks)):
        if masks[i][0][0][0] == 0:
            delet.append(i)
    return delet

# Ïðîáåãàåò ïî ïåðåäàííûì ïàðàìåòðàì ìîäåëè (âïåðåä èëè íàçàä îïðåäåëÿåòñÿ èíâåðòèðîâàííûì ìàñèâîâ ïàðàìåòðîâ),
# íàõîäèò ñëîé êîòîðûé íóæíî óäàëèòü Delet_Name_sloi è óäàëÿåò â íåì è ñëåäóþõèõ ñòðîêè Delet_indeks.
# Îñòàíàâëèâàåòñÿ, êîãäà äîøåë äî êîíâàëþöèîííîãî ñëîÿ ñ groups == 1 è â íåì óäàëÿåò ñòîëáåö 
# ïðè äâèæåíèè âïåðåä è ñòðîêó ïðè äâèæåíèè íàçàä. Íàïðàâëåíèå çíàåò ïî ïàðàìåòðó obratno
def go_model(model, names, Delet_Name_sloi, Delet_indeks, obratno = False, priint = False):
    is_delett = False                                              # Â ðåæèìå óäàëåíèÿ
    for name in names:                                             # Ïðîéòèñü ïî ñëîÿì ìîäåëè ñ ïàðàìåòðàìè
        Name_the_sloi = rename(name[0].split(".weight")[0])        # Íàçâàíèå òåêóùåãî ñëîÿ
        not_bias = len(name[0].split(".bias")) == 1                # ×òîáû íå ïîðåçàòü äâàæäû 1 ñëîé
        if Delet_Name_sloi == name[0].split(".weight")[0]:         # Íàøëè ñëîé, êîòîðûé íóæíî óäàëèòü
            is_delett = True
            if obratno:
                continue                                           # Òåêóùèé ñäëé óæå áûë ïîðåçàí ïðè äâèæåíèè âïåðåä
        # Åñëè òåêóùèé ñòîé - ñëîé êîíâàëþöèè è ìû â ðåæèìå óäàëåíèÿ, è ýòîò ïàðàìåòð íå bias
        elif str(type(eval("model.{}".format(Name_the_sloi)))).split("'")[1] == \
        'torch.nn.modules.conv.Conv2d' and is_delett and not_bias:
            if eval("model.{}".format(Name_the_sloi)).groups == 1: # Ýòîò ñëîé íå ñîñòîèò èç äóáëèêàòîâ ñòîëáöîâ
                is_delett = False                                  # Âûõîäèì èç ðåæèìà óäàëåíèÿ
                if not priint:                                     # Îòîáðàæàåì èëè óäàëÿåì, â çàâèñèìîñòè îò ðåæèìà ðàáîòû
                    if obratno:
                        delet(model, Name_the_sloi, i = Delet_indeks, j = -1)
                    else:
                        delet(model, Name_the_sloi, i = -1, j = Delet_indeks)
                else:
                    print("K",name)
        if is_delett and not_bias:                                 # Óäàëÿåì, åñëè ìû â ðåæèìå óäàëåíèÿ è ýòîò ïàðàìåòð íå bias
            if not priint:
                delet(model, Name_the_sloi, i = Delet_indeks, j = -1)
            else:
                print("S",name)                                    # Èëè ïðèíòóåì, åñëè òàêîâ ðåæèì ðàáîòû

# Ïðîáåãàåò ïî ïåðåäàííûì ïàðàìåòðàì ìîäåëè (âïåðåä èëè íàçàä îïðåäåëÿåòñÿ èíâåðòèðîâàííûì ìàñèâîâ ïàðàìåòðîâ),
# íàõîäèò ñëîé êîòîðûé íóæíî óäàëèòü Delet_Name_sloi, çàïîìèíàåò åãî ðàçìåðíîñòü
# è óäàëÿåò â íåì è âñåõ ñëåäóþõèõ ñëîÿõ ñ ýòîé ðàçìåðíîñòüþ ñòðîêè è ñòîëáöû Delet_indeks.
# Îñòàíàâëèâàåòñÿ, åñëè stop = True è êîãäà äîøåë äî êîíâàëþöèîííîãî ñëîÿ èç äðóãîâî áëîêà
# ïðè äâèæåíèè âïåðåä óäàëÿåì â íåì ñòîëáåö à ïðè äâèæåíèè íàçàä íè÷åãî. Íàïðàâëåíèå çíàåò ïî ïàðàìåòðó obratno
def go_model_ofa_scip(model, names, Delet_Name_sloi, Delet_indeks, obratno = False, priint = False, stop = True):
    is_delett = False                                              # Â ðåæèìå óäàëåíèÿ
    Rename_Delet_Name_sloi = rename(Delet_Name_sloi)               # Îáðàùåíèå ê ñëîþ
    size = -1
    for name in names:                                             # Ïðîéòèñü ïî ñëîÿì ìîäåëè ñ ïàðàìåòðàìè
        not_bias = len(name[0].split(".bias")) == 1                # ×òîáû íå ïîðåçàòü äâàæäû 1 ñëîé
        Name_the_sloi = rename(name[0].split(".weight")[0])        # Íàçâàíèå òåêóùåãî ñëîÿ
        if Delet_Name_sloi == name[0].split(".weight")[0]:         # Íàøëè ñëîé, êîòîðûé íóæíî óäàëèòü
            is_delett = True                                       # Ïåðåéòè â ðåæèì óäàëåíèÿ
            size = name[2][-1]                                     # Çàïîìíèòü ðàçìåðíîñòü óäîëÿåìîãî ñëîÿ
            if obratno:                                            # Åñëè ìû óäàëÿåì äâèãàÿñü íàçàä ïî ñåòè
                size = size + len(Delet_indeks)                    # âîñòîíàâëèâàåì èíôîðìàöèþ î ðàçìåðíîñòè
                continue                                           # âåäü òåêóùèé ñëîé óæå ïîðåçàí
        elif stop:                                                 # Ìû óäàëÿåì ïî çàêîíîìåðíîñòÿì îôû
            # Ýòî äðóãîé áëîê è ìû â ðåæèìå óäàëåíèÿ
            if is_delett and Rename_Delet_Name_sloi.split("[")[1].split("]")[0] !=  Name_the_sloi.split("[")[1].split("]")[0]:
                if obratno:                         # Ïðè äâèæåíèè íàçàä
                    is_delett = False               # óäàëÿòü ïîñëåäíèé ýëåìåíò áëîêà â êîíöå íå íóæíî
                    break             
                elif name[1] == 'torch.nn.modules.conv.Conv2d': # Ïðè äâèæåíèè âïåðåä, åñëè ýòîò ñëîé - ñëîé êîíâîëþöèè
                    is_delett = False                           # Âûõîäèì èç ðåæèìà óäàëåíèÿ
                    if name[2][-1] == size or (name[2][0] == size  and name[1] == 'torch.nn.modules.linear.Linear'): # Åñëè ó ïåðâîãî ýëåìåíòà ñëåäóþùåãî ñëîÿ ðàçìåðíîñòü êàê ó óäîëÿåìîãî
                        if not priint:                          # Óäîëÿåì èëè ïðèíòóåì â çàâèñèìîñòè îò ðåæèìà ðàáîòû
                            if obratno:
                                print("K",name)
                                delet(model, Name_the_sloi, i = Delet_indeks, j = -1)
                            else:
                                print("K",name)
                                delet(model, Name_the_sloi, i = -1, j = Delet_indeks)
                        else:
                            print("K",name)
        if is_delett and not_bias:              # Åñëè ìû â ðåæèìå óäàëåíèÿ è ýòîò ïàðàìåòð íå bias
            if name[2][-1] == size or (name[2][0] == size  and name[1] == 'torch.nn.modules.linear.Linear'): # Ðàçìåðíîñòü ñòðîê ñîâïàäàåò (êîíâîëþöèè è áà÷íîðì) ñ îáçåçàåìûì ñëîåì
                if not priint:                  # Óäîëÿåì èõ åñëè ìû â ðåæèìå óäàëåíèÿ
                    print("S",name)
                    delet(model, Name_the_sloi, i = Delet_indeks, j = -1)
                else:
                    print("S",name)
            if name[2][0] == size and name[1] == 'torch.nn.modules.conv.Conv2d': # Ñòîëáöû êîíâîëþöèîííûõ ìàòðèö ñ ýòîé ðàçìåðíîñòüþ
                if eval("model.{}".format(Name_the_sloi)).groups==1:             # êîòîðûå íå ñîñòîÿò èç äóáëåêàòîâ ñåáÿ
                    if not priint:                                               # òòîæå óäàëÿåì èëè ïðèíòóåì
                        print("S",name)
                        delet(model, Name_the_sloi, i = -1, j = Delet_indeks)
                    else:
                        print("S",name)

# Ïðîõîäåò ïî ìàñêå è çàïóñêàåò óäàëåíèå ñòðîê ñëîÿ èç ìàñêè è ñâÿçàííûõ ñ íèì ñëî¸ì
def compres(model, masks):
    for i in masks:
        for j in masks[i]:
            
            Name = rename(i)                       # Îáðàùåíèå ê ñëîþ
            Delet_indeks = get_delet(masks[i][j])  # ïîëó÷èòü èíäåêñû ñòðîê äëÿ óäàëåíèÿ èç ìàñêè ïàðàìåòðîâ
            # Óäàëèòü êîìïîíåíòû ñëîÿ ïî ìàñêå
            go_model(model, get_stract(model), i, Delet_indeks, obratno = False)
            # Åñëè ýòî êâàäðàòíàÿ ìàòðèöà
            if eval("model.{}".format(Name)).in_channels == eval("model.{}".format(Name)).out_channels:
                # Òî óäàëÿòü è â îáðàòíóþ ñòîðîíó
                go_model(model, obr(get_stract(model)), i, Delet_indeks, obratno = True)
            # Ïîâòîðèòü íî òîëüêî îòîáðàæàòü êàêèå ñëîè ðåçàëèòü    
            go_model(model, get_stract(model), i, Delet_indeks, obratno = False, priint = True)
            if eval("model.{}".format(Name)).in_channels == eval("model.{}".format(Name)).out_channels:
                go_model(model, obr(get_stract(model)), i, Delet_indeks, obratno = True, priint = True)

# Ïðîõîäåò ïî ìàñêå è çàïóñêàåò óäàëåíèå ñòðîê ñëîÿ èç ìàñêè è ñâÿçàííûõ ñ íèì ñëî¸ì
def compres2(model, masks, ofa = False, stop = True):
    for i in masks:
        for j in masks[i]:
            Name = rename(i)    # Îáðàùåíèå ê ñëîþ
            print("d:",Name)
            # Óäàëåíèå ñòàíäàðòíîé ìîäåëè, ïî òèïó mobilenetv2
            if not ofa:
                Delet_indeks = get_delet(masks[i][j])
                go_model(model,get_stract(model), i, Delet_indeks, obratno = False)
                if eval("model.{}".format(Name)).in_channels == eval("model.{}".format(Name)).out_channels:
                    go_model(model,obr(get_stract(model)), i, Delet_indeks, obratno = True)
                go_model(model,get_stract(model), i, Delet_indeks, obratno = False, priint = True)
                if eval("model.{}".format(Name)).in_channels == eval("model.{}".format(Name)).out_channels:
                    go_model(model,obr(get_stract(model)), i, Delet_indeks, obratno = True, priint = True)
            # Óäàëåíèå ìîäåëè, ñ îñîáåííîñòÿìè ofa
            elif stop:
                Delet_indeks = get_delet(masks[i][j])
                go_model_ofa_scip(model,get_stract(model), i, Delet_indeks, obratno = False)
                go_model_ofa_scip(model,obr(get_stract(model)), i, Delet_indeks, obratno = True)
                go_model_ofa_scip(model,get_stract(model), i, Delet_indeks, obratno = False, priint = True)
                go_model_ofa_scip(model,obr(get_stract(model)), i, Delet_indeks, obratno = True, priint = True)
            # 100% ðàáîòàþùåå óäàëåíèå ñëîåâ ìîäåëè, åñëè àñîáåííîñòè ñëèøêîì òÿæîëûå
            else:
                Delet_indeks = get_delet(masks[i][j])
                go_model_ofa_scip(model,get_stract(model), i, Delet_indeks, obratno = False, stop = False)
                go_model_ofa_scip(model,obr(get_stract(model)), i, Delet_indeks, obratno = True, stop = False)
                go_model_ofa_scip(model,get_stract(model), i, Delet_indeks, obratno = False, priint = True, stop = False)
                go_model_ofa_scip(model,obr(get_stract(model)), i, Delet_indeks, obratno = True, priint = True, stop = False)

# Ïîëó÷èòü ïåðåäàííûå ïàðàìåòðû
def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_ind', type=int, default=0, help='Ïîðÿäêîâûé íîìåð ðàáîòíèêà. ×òîáû îòëè÷àòü èõ ìåæäó ñîáîé')
    parser.add_argument('--cuda', type=int, default=0, help='Íîìåð êàðòû')
    parser.add_argument('--load', type=str,  help='Ïóòü ê ìîäåëè â ôîðìàòå ñîâìåñòèìûì ñ "model = torch.load(load)"')
    parser.add_argument('--name_sloi', type=str,  help='Íàçâàíèå ñëîÿ äëÿ óäàëåíèÿ')
    parser.add_argument('--sparsity', type=float, default=0.1, help='% êîòîðûé íóæíî îáðåçàòü îò ñëîÿ')
    parser.add_argument('--orig_size', type=float, default=100, help=' ')
    parser.add_argument('--iterr', type=int, default=0, help='Èòåðàöèÿ àëãîðèòìà îáðåçêè, íóæåí äëÿ ëîãîâ')
    parser.add_argument('--algoritm', type=str, default=0, help='Íèçêîóðîâíåâûé àëãîðèòì îáðåçêè. Âûáîð èç: TaylorFOWeight, L2Norm')
    param = parser.parse_args()
    return param

# ×àñòü àëãîðèòìà âûíåñåííàÿ â ô-þ, ÷òîáû íå äóáëèðîâàòü êîä.
# Ïîíÿòü ÷òî ïðóíèíã íåóäàëñÿ ìîæíî òîëüêî çàïóñòèâ îáó÷åíèå è óâèäèâ îøèáêó.
# Çàïóñê ïðóíèíãà è ïîñëåäóþùåå äîîáó÷åíèå ñ çàïèñüþ ëîãîâ â ôàéë.
def pruning_type(model, masks, do, param, config_list, type_pruning = "defolt"):
    import yaml
    import torch.optim as optim
    config = yaml.safe_load(open('Pruning.yaml'))
    import cod.trainer as trainer
    
    alf        = config['my_pruning']['alf']
    lr         = config['retraining']['lr']
    num_epochs = config['retraining']['num_epochs']
    snp        = config['path']['exp_save'] + "/" + config['path']['model_name']
    modelName  = config['path']['model_name']
    fil_it = snp + "/" + modelName + "_it{}.txt".format(param.iterr)
    model = torch.load(param.load)                                       # Çàãðóçêà ìîäåëè (ìîäåëü îñòàëàñü ïîðåçàííîé)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(type_pruning)
    if do[0]*do[1] == param.orig_size:
        if type_pruning == "defolt":
            compres2(model, masks)                                           # Çàïóñê ïðóíèíãà
        elif type_pruning == "ofa":
            compres2(model, masks, ofa = True)                               # Çàïóñê ïðóíèíãà
        elif type_pruning == "total":
            compres2(model, masks, ofa = True, stop = False)                 # Çàïóñê ïðóíèíãà
    
    model.to(device)                                                     # Çàìåíåííûå ñëîè íå íà êàðòå
    # Óçíàòü ðàçìåð ñâåðòîê ïîñëå ïðóíèíãà
    for name in get_stract(model):
        if name[0].split(".weight")[0] == config_list[0]['op_names'][0]:
            posle = name[2]
            break

    # Ïîñëå îáðåçêè ñåòü îáðåçñëàñü è êðàòíà alf
    
    if do != posle and posle[1] % alf == 0:                       
        # Äîîáó÷àåì
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model, loss, acc, st, time_elapsed2 =trainer.trainer(model, optimizer, trainer.criterion, num_epochs = num_epochs, ind = param.self_ind)
        # Çàïèñü ðåçóëüòàòà
        f = open(fil_it, "a")
        strok = str(param.self_ind) + " " + config_list[0]['op_names'][0] + " " + \
        str(acc) +" "+ str(do) + " "+ str(posle)+ " " + "type_pruning " + type_pruning + "\n"
        f.write(strok)
        f.close()
        # Ñîõðàíåíèå ìîäåëè
        torch.save(model, snp + "/" + modelName + "_" + config_list[0]['op_names'][0] + \
                   "_it_{}_acc_{:.3f}.pth".format(param.iterr,acc))
    else:
        # Çàïèñü ðàçìåðíîñòåé äî è ïîñëå ïðóíèíãà
        f = open(fil_it, "a")
        strok = str(param.self_ind) + " " + config_list[0]['op_names'][0] + " " + \
        "EROR" +" "+ str(do) + " "+ str(posle)+ " " +"type_pruning " + type_pruning + "\n"
        f.write(strok)
        f.close()
            
def main(param):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(param.cuda)
    import torch
    import nni
    from nni.algorithms.compression.v2.pytorch.pruning import TaylorFOWeightPruner
    from nni.algorithms.compression.v2.pytorch.pruning import L2NormPruner
    from nni.compression.pytorch import ModelSpeedup
    import yaml
    import cod.trainer as trainer
    # Ïàðàìåòðû ïðóíèíãà nni
    config_list = [{'sparsity': param.sparsity, 
                'op_types': ['Conv2d'],
                'op_names': [param.name_sloi]
               }]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(param.load)
    model = model.to(device)
    traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
    
    # Çàïîìèíàíèå ðàçìåðíîñòè ñëîÿ äî ïðóíèíãà
    do = posle = 0
    for name in get_stract(model):
        if name[0].split(".weight")[0] == config_list[0]['op_names'][0]:
            do = name[2]
    # Âûáîð àëãîðèòìà ïðóíèíãà
    pruner = None
    if param.algoritm == "TaylorFOWeight":
        pruner = TaylorFOWeightPruner(model, config_list, trainer.trainer, 
                                      traced_optimizer, trainer.criterion, 
                                      training_batches = trainer.batch_size_t)
    elif param.algoritm == "MeanRank":
        pruner = ActivationMeanRankPruner(model, config_list, trainer.trainer, 
                                          traced_optimizer, trainer.criterion, 
                                          training_batches = trainer.batch_size_t)
    elif param.algoritm == "L2Norm":
        pruner = L2NormPruner(model, config_list)
    
    # Çàïóñê ïðóíèíãà îò nni
    model, masks = pruner.compress()
    pruner._unwrap_model()
    type_pruning = ""
    # Îáðåçêà ñåòè íà îñíîâå ìàñêè îò nni
    try:
        # Èäåàëüíàÿ îáðåçêà (òîëüêî ñâÿçàííûõ ñëî¸â)
        pruning_type(model, masks, do, param, config_list, type_pruning = "defolt")
    except:
        try:
            # Îáðåçêà ïî îñîáåííîñòÿì ofa
            pruning_type(model, masks, do, param, config_list, type_pruning = "ofa")
        except:
            # Êðàéíå ïëîõàÿ, íî òî÷íî ðàáîòàþùàÿ îáðåçêà
            pruning_type(model, masks, do, param, config_list, type_pruning = "total")
                
if __name__ == "__main__":
    param = get_param()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(param.cuda)
    main(param)
