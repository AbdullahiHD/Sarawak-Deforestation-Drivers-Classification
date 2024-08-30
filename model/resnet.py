# ResNet18 custom definition from N. Ramachandran et al., "Automatic deforestation driver attribution using deep learning on satellite imagery," Global Environmental Change,https://arxiv.org/abs/2011.05479 accessed from https://github.com/bellasih/multimodal_supercon through https://paperswithcode.com/paper/forestnet-classifying-drivers-of#code
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
from torchvision import (
    transforms,
    utils,
    models,
    transforms,
)  # Duplicate import of transforms
import torch.optim as optim
from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
from torch.hub import (
    load_state_dict_from_url,
)  


# Define a custom ResNet-based model class with additional layers for MLP
class ResNetMLP(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNetMLP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Initial settings for the model
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            1024 * block.expansion, num_classes
        )  # Final fully connected layer
        self.softmax = nn.Softmax(
            dim=num_classes
        )  # Softmax activation for classification

        # Additional layers for the slope input (MLP part)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(160 * 160, 512)
        self.linear2 = nn.Linear(512, 512)
        self.batch1d = nn.BatchNorm1d(512)
        self.batch1d_n = nn.BatchNorm1d(160 * 160)
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last batch normalization layer in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Helper function to create a layer of the ResNet
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _del_fc(self):
        # Function to delete the fully connected layer
        self.fc = None

    def _forward_impl(self, x, s):
        # Forward pass implementation
        # Processing the image input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Processing the slope input
        s = self.flatten(s)
        s = self.batch1d_n(s)
        s = self.dropout(s)
        s = self.batch1d(F.leaky_relu(self.linear1(s)))
        s = F.leaky_relu(self.linear2(s))

        # Concatenate image and slope features and pass through the final fully connected layer
        x = torch.concat((x, s), 1)
        x = self.fc(self.dropout(x))

        return x

    def forward(self, x, s):
        # Forward method for the model
        return self._forward_impl(x, s)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # Helper function to create a ResNet model
    model = ResNetMLP(block, layers, **kwargs)
    if pretrained:
        # Load pre-trained weights if specified
        checkpoint = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, projection=True, **kwargs):
    # Function to create a ResNet-18 model
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
