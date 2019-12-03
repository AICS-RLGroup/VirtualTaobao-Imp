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

    @staticmethod
    def softmax_feature(x):
        features = [None] * 11
        features[0] = x[:, 0:8]
        features[1] = x[:, 8:16]
        features[2] = x[:, 16:27]
        features[3] = x[:, 27:38]
        features[4] = x[:, 38:49]
        features[5] = x[:, 49:60]
        features[6] = x[:, 60:62]
        features[7] = x[:, 62:64]
        features[8] = x[:, 64:67]
        features[9] = x[:, 67:85]
        features[10] = x[:, 85:88]
        entropy = 0
        softmax_feature = FLOAT([])
        for i in range(11):
            softmax_feature = torch.cat((softmax_feature, F.softmax(features[i], dim=1)), dim=-1)
            entropy += -(F.log_softmax(features[i], dim=1) * F.softmax(features[i], dim=1)).sum(dim=1).mean()
        return softmax_feature, entropy