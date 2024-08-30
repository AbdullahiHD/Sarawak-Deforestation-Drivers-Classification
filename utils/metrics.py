# Custom metrics definition from N. Ramachandran et al., "Automatic deforestation driver attribution using deep learning on satellite imagery," Global Environmental Change,https://arxiv.org/abs/2011.05479 accessed from https://github.com/bellasih/multimodal_supercon through https://paperswithcode.com/paper/forestnet-classifying-drivers-of#code
# @article{10.1117/1.JRS.17.036502,
# author = {Bella Septina Ika Hartanti and Valentino Vito and Aniati Murni Arymurthy and Adila Alfa Krisnadhi and Andie Setiyoko},
# title = {{Multimodal SuperCon: classifier for drivers of deforestation in Indonesia}},
# volume = {17},
# journal = {Journal of Applied Remote Sensing},
# number = {3},
# publisher = {SPIE},
# pages = {036502},
# keywords = {deforestation driver classification, contrastive learning, class imbalance, multimodal fusion, Machine learning, Education and training, Data modeling, Image fusion, Performance modeling, Atmospheric modeling, Data fusion, Deep learning, Landsat, RGB color model},
# year = {2023},
# doi = {10.1117/1.JRS.17.036502},
# URL = {https://doi.org/10.1117/1.JRS.17.036502}
# }


import torch
import torch.nn as nn
import torch.nn.functional as F

def get_acc_seg(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = (outputs==segs)
    acc = acc.view(-1)
    return acc.sum()/len(acc)

def get_acc_seg_weighted(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = []
    for i in range(5):
        acc_temp = (outputs==segs)
        acc_temp = acc_temp.view(-1)
        acc.append(acc_temp.sum()/len(acc_temp))
    return torch.mean(torch.stack(acc))

def get_acc_nzero(outputs, segs):
    mask = ~segs.eq(0)
    outputs = torch.max(outputs, dim=1)[1]
    acc = torch.masked_select((outputs==segs), mask)
    return acc.sum()/len(acc)

def get_acc_class(outputs, labels):
    outputs = torch.max(outputs, dim=1)[1]
    acc = (outputs==labels)
    return acc.sum()/len(acc)

def get_acc_binseg(outputs, segs):
    outputs = F.sigmoid(outputs)
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    acc = (outputs==segs)
    acc_1 = acc[segs==1].view(-1)
    acc_0 = acc[segs==0].view(-1)
    acc_1 = acc_1.sum()/len(acc_1)
    acc_0 = acc_0.sum()/len(acc_0)
    return acc_1, acc_0, (acc_1+acc_0)/2
