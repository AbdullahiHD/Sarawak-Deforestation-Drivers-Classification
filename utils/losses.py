# custom losses definition from N. Ramachandran et al., "Automatic deforestation driver attribution using deep learning on satellite imagery," Global Environmental Change,https://arxiv.org/abs/2011.05479 accessed from https://github.com/bellasih/multimodal_supercon through https://paperswithcode.com/paper/forestnet-classifying-drivers-of#code
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
from pytorch_metric_learning import losses

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, alpha = 0.8, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha 

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()
        return focal_loss

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))
