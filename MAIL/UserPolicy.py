#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/4 下午7:46

from utils.utils import *


class UserPolicy(nn.Module):
    def __init__(self, n_input=88 + 27 + 1, n_output=11, activation=nn.LeakyReLU):
        super(UserPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            activation(),
            nn.Linear(128, 256),
            activation(),
            nn.Linear(256, n_output)
        )
        self.model.apply(init_weight)

    def forward(self, user_feature, engine_action, page_index):
        x = self.model(torch.cat((user_feature, engine_action, page_index), dim=-1))
        action = torch.multinomial(F.softmax(x, dim=1), 1)

        return action
