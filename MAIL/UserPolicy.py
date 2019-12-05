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

    def forward(self, x):
        action_prob = F.softmax(self.model(x), dim=1)
        return action_prob

    def get_action(self, x):
        # x = torch.cat((user_feature, engine_action, page_index), dim=-1)
        action_prob = self.forward(x)
        action = torch.multinomial(action_prob, 1)
        return action

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))
