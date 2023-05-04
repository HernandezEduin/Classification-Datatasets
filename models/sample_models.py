# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:03:38 2022

@author: Eduin Hernandez
"""

from collections import OrderedDict

import torch.nn as nn
from models.resnet import ResNet
from models.vggnet import VGG

class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(LinearRegression, self).__init__()
        
        self.flat = nn.Flatten()
        self.dense = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x):                
        x = self.flat(x)
        x = self.dense(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int, layers: int, bias: bool = True,
                 relu: bool = False, dropout: bool = False):
        super(MLP, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.flat = nn.Flatten()
        self.model = nn.Sequential(self._create(in_features, out_features, hidden, layers, bias, dropout))

    def _create(self, in_features, out_features, hidden, layers, bias = True, relu = False, dropout = False):
        if layers == 1:
            d = OrderedDict()
            d['linear0'] = nn.Linear(in_features, out_features, bias=bias)
            return d
        
        d = OrderedDict()
        for i in range(layers):
            if i == 0:
                d['linear' + str(i)] = nn.Linear(in_features, hidden, bias=bias)
                if relu:
                    d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
            elif i == layers - 1:
                d['linear' + str(i)] = nn.Linear(hidden, out_features, bias=bias)
            else:
                d['linear' + str(i)] = nn.Linear(hidden, hidden, bias=bias)
                if relu:
                    d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
        return d
    
    def forward(self, x):
        x = self.flat(x)
        x = self.model(x)
        return x

class LeNet(nn.Module):
    def __init__(self, bias: bool = False):
        super(LeNet, self).__init__()        
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2, bias=bias)    # output becomes 28x28
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0, bias=bias)   # output becomes 10x10

        self.pool = nn.MaxPool2d(2, 2)

        self.dense1 = nn.Linear(16*5*5,120, bias=bias)
        self.dense2 = nn.Linear(120,84, bias=bias)
        self.dense3 = nn.Linear(84,10, bias=bias)
        
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
    
    def forward(self, x):
        y = self.pool(self.relu(self.conv1(x)))
        y = self.pool(self.relu(self.conv2(y)))
        
        y = self.flat(y)
        y = self.relu(self.dense1(y))
        y = self.relu(self.dense2(y))
        y = self.dense3(y)
        
        return y
    
#------------------------------------------------------------------------------
models_dict = {'LinearRegression': LinearRegression,
               'MLP': MLP,
               'LeNet': LeNet,
               'ResNet': ResNet,
               'VGG': VGG}