import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels

import pdb


class MainModel(nn.Module):
    def __init__(self, backbone, use_dcl=True, numcls=4, use_Asoftmax=False):
        super(MainModel, self).__init__()
        self.use_dcl = use_dcl
        self.num_classes = numcls
        self.use_Asoftmax = use_Asoftmax

        # self.model = backbone
        # print(nn.Sequential(*list(backbone.children())[:-2]))
        self.model = nn.Sequential(*list(backbone.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        if self.use_dcl:
            self.classifier_swap = nn.Linear(2048, 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        return out