#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午6:40

from utils.utils import *


class GeneratorModel(nn.Module):
    def __init__(self, dim_user=88, dim_seed=128, dim_hidden=128, activation=nn.ReLU):
        super(GeneratorModel, self).__init__()
        self.dim_seed = dim_seed
        self.generator = nn.Sequential(
            nn.Linear(dim_seed, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_user),
        )
        self.generator.apply(init_weight)

    def forward(self, z):
        return self.generator(z)
