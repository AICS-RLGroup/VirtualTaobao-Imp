#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/5 下午2:59
from GAN_SD.GeneratorModel import GeneratorModel
from utils.utils import *


class MailPolicy(nn.Module):
    def __init__(self, activation=nn.LeakyReLU):
        super(MailPolicy, self).__init__()

        self.dim_user_state = 88 + 27 + 1
        self.dim_user_action = 11

        self.dim_engine_state = 88
        self.dim_engine_hidden = 256
        self.dim_engine_action = 27

        self.dim_userleave_state = 88
        self.dim_userleave_action = 101

        self.UserModel = GeneratorModel()
        self.UserModel.load()

        self.EnginePolicy = nn.Sequential(
            nn.Linear(self.dim_engine_state, self.dim_engine_hidden),
            activation(),
            nn.Linear(self.dim_engine_hidden, self.dim_engine_action),
        )

        self.UserPolicy = nn.Sequential(
            nn.Linear(self.dim_user_state, 128),
            activation(),
            nn.Linear(128, 256),
            activation(),
            nn.Linear(256, self.dim_user_action)
        )

        self.UserLeavePolicy = nn.Sequential(
            nn.Linear(self.dim_user_feature, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.dim_userleave_action)
        )

        self.UserPolicy.apply(init_weight)
        self.EnginePolicy.apply(init_weight)
        self.UserLeavePolicy.apply(init_weight)

    def get_engine_action(self, engine_state):
        return self.EnginePolicy(engine_state)

    # user_state (user_feature, engine_action, page_index)
    def get_user_action_prob(self, user_state):
        action_prob = F.softmax(self.UserPolicy(user_state), dim=1)
        return action_prob

    def get_user_action(self, user_state):
        action_prob = self.get_user_action_prob(user_state)
        action = torch.multinomial(action_prob, 1)
        return action, action_prob

    def get_user_leave_action(self, user):
        x = self.UserLeavePolicy(user)
        leave_page_index = torch.multinomial(F.softmax(x, dim=1), 1)
        return leave_page_index

    def generate_trajectory(self, trajectory_num):
        # sample J trajectories
        trajectory = []
        for j in range(trajectory_num):
            tao_j = []
            # sample user from GAN-SD distribution
            s, _ = self.UserModel.generate()
            # get user's leave page index from leave model
            leave_page_index = self.get_user_leave_action(s)
            # get engine action from user with request
            a = self.EnginePolicy(s)

            s_c = torch.cat((s, a), dim=1)
            page_index = 1

            while page_index != leave_page_index:  # terminate condition
                s_c = torch.cat((s_c, FLOAT(page_index)).to(device), dim=1)
                a_c = self.UserPolicy.get_action(s_c)
                tao_j.append(torch.cat((s_c, a_c), dim=1))

                # genreate new customer state
                a = self.EnginePolicy(s)
                s_c = torch.cat((s, a))

                page_index += 1

            trajectory.append(tao_j)
        return trajectory

    def get_log_prob(self, user_state, user_action):
        _, action_prob = self.get_user_action(user_state)
        current_action_prob = action_prob[:, user_action]

        return torch.log(current_action_prob.unsqueeze(1))
