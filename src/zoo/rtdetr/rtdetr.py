"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone            #Presnet
        self.decoder = decoder              #HybridEncoder
        self.encoder = encoder              #RTDETRTransformer
        
    def forward(self, x, targets=None): #targets标注,字典列表，每个字典keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'idx'])
        x = self.backbone(x)            #特征图列表，下采样8,16,32
        x = self.encoder(x)             #编码器+PAN交互过的特征图列表，尺度从大到小，c=256
        x = self.decoder(x, targets)    #

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
