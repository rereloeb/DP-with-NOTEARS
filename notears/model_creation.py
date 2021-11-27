from locally_connected import LocallyConnected
from trace_expm import trace_expm

import torch
import torch.nn as nn
import numpy as np
import math
import ipdb
import epsilon_calculation
import utils as ut
import batch_samplers
import argparse



class NotearsMLP1(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP1, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        #stdv = math.sqrt( 1.0 / d )
        #nn.init.uniform_(self.fc1_pos.weight.data, 0, stdv)
        #nn.init.uniform_(self.fc1_neg.weight.data, 0, stdv)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        print("model created with box penalty")

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def bound_penalty(self):
        zeros = torch.zeros_like(self.fc1_pos.weight)
        t1 = torch.maximum(-self.fc1_pos.weight,zeros)
        t2 = torch.maximum(-self.fc1_neg.weight,zeros)
        u1 = torch.square(t1)
        u2 = torch.square(t2)
        pen = u1.sum()+u2.sum()
        #print(pen)
        return pen

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        #print("Adj matrix[i,j] = 2-norm squared of column i in first layer of NN j = ",A)
        #h = trace_expm(A) - d #(Zheng et al. 2018)
        #A different formulation, slightly faster at the cost of numerical stability (Yu et al. 2019)
        alpha = 1/(d*1.0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        M = torch.eye(d).to(device) + A.to(device) *alpha
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W



class NotearsMLP2(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP2, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
#       self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        #stdv = math.sqrt( 1.0 / d )
        #nn.init.uniform_(self.fc1_pos.weight.data, 0, stdv)
        #nn.init.uniform_(self.fc1_neg.weight.data, 0, stdv)
#        self.fc1_pos.weight.bounds = self._bounds()
#        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        print("model created without box penalty")

#    def _bounds(self):
#        d = self.dims[0]
#        bounds = []
#        for j in range(d):
#            for m in range(self.dims[1]):
#                for i in range(d):
#                    if i == j:
#                        bound = (0, 0)
#                    else:
#                        bound = (0, None)
#                    bounds.append(bound)
#        return bounds

#    def bound_penalty(self):
#        zeros = torch.zeros_like(self.fc1_pos.weight)
#        t1 = torch.maximum(-self.fc1_pos.weight,zeros)
#        t2 = torch.maximum(-self.fc1_neg.weight,zeros)
#        u1 = torch.square(t1)
#        u2 = torch.square(t2)
#        pen = u1.sum()+u2.sum()
#         print(pen)
#         return pen

    def forward(self, x):  # [n, d] -> [n, d]
#        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = self.fc1_pos(x)
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
#        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = self.fc1_pos.weight
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        #print("Adj matrix[i,j] = 2-norm squared of column i in first layer of NN j = ",A)
        #h = trace_expm(A) - d #(Zheng et al. 2018)
        #A different formulation, slightly faster at the cost of numerical stability (Yu et al. 2019)
        alpha = 1/(d*1.0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        M = torch.eye(d).to(device) + A.to(device) *alpha
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
#        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = self.fc1_pos.weight
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
#        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        fc1_weight = torch.abs(self.fc1_pos.weight)
        reg = torch.sum(fc1_weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
#        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = self.fc1_pos.weight
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def NotearsMLP(dims, boxpenalty, bias):
    if boxpenalty:
        return NotearsMLP1(dims,bias)
    else:
        return NotearsMLP2(dims,bias)


