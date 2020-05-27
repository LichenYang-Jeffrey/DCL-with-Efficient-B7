import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class SupConNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, backbone, head='mlp', dim_in=2048, feat_dim=128):
        super(SupConNet, self).__init__()
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat