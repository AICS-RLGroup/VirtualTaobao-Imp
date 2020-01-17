#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午4:17
import random

from torch.distributions import Normal, OneHotCategorical

from utils.replay_memory import Memory
from utils.utils import *

# 离散 action 对应
action_sizes = [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]
n_actions = len(action_sizes)
action_sizes_cum = np.cumsum(action_sizes)
action_slices = [slice(action_sizes_cum[i] - action_sizes[i], action_sizes_cum[i]) for i in
                 range(n_actions)]


# (s, a) -> s'
class EnvPolicy(nn.Module):
    def __init__(self, dim_state_continuous=23, dim_state_disc=132, dim_action=6, dim_hidden=256,
                 activation=nn.LeakyReLU):
        super(EnvPolicy, self).__init__()

        self.dim_state_continuous = dim_state_continuous
        self.dim_state_disc = dim_state_disc
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.dim_state = self.dim_state_disc + self.dim_state_continuous

        self.common = nn.Sequential(
            nn.Linear(self.dim_state + self.dim_action, self.dim_hidden),
            activation()
        )

        self.action_disc = nn.Linear(self.dim_hidden, self.dim_state_disc)
        self.action_continuous = nn.Linear(self.dim_hidden, self.dim_state_continuous * 2)

    def forward(self, x):
        """
        :param x: (state, action)
        :return: next_state
        """
        x = self.common(x)
        action_disc = self.action_disc(x)
        action_continuous = self.action_continuous(x)
        action_continuous_mean, action_continuous_std = action_continuous[:, :self.dim_state_continuous], \
                                                        action_continuous[:, self.dim_state_continuous:]
        action_continuous_mean = torch.clamp(action_continuous_mean, -1, 1)
        action_continuous_std = torch.clamp(action_continuous_std, 0, 1)
        return action_disc, action_continuous_mean, action_continuous_std

    @staticmethod
    def get_softmax_disc_action(x):
        # 15个one-hot向量拼接而成，大小分别是
        action_disc = [None] * n_actions
        for i in range(n_actions):
            action_disc[i] = x[:, action_slices[i]]

        soft_max_action = FLOAT([]).to(device)
        one_hot_action = FLOAT([]).to(device)

        for i in range(n_actions):
            soft_max_action = torch.cat((soft_max_action, F.softmax(action_disc[i], dim=1).to(device)), dim=-1)
            tmp = torch.zeros_like(action_disc[i]).to(device)
            one_hot_action = torch.cat(
                (one_hot_action, tmp.scatter_(1, torch.multinomial(F.softmax(action_disc[i], dim=1), 1), 1)),
                dim=-1)  # 根据softmax feature 生成one-hot的feature
        return soft_max_action, one_hot_action

    def get_action(self, x):
        action_disc, action_continuous_mean, action_continuous_std = self.forward(x)
        softmax_disc_action, action_disc = self.get_softmax_disc_action(action_disc)
        normal_dist = Normal(action_continuous_mean, action_continuous_std)
        action_continuous = normal_dist.rsample()

        action = torch.cat([action_disc, action_continuous], dim=-1)
        return action


# s -> a
class AgentPolicy(nn.Module):
    def __init__(self, dim_state=155, dim_action=6, dim_hidden=128,
                 activation=nn.LeakyReLU):
        super(AgentPolicy, self).__init__()

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden

        self.common = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation()
        )

        self.action = nn.Linear(self.dim_hidden, self.dim_action * 2)

    def forward(self, x):
        """
        :param x: state
        :return: action
        """
        x = self.common(x)
        action = self.action(x)
        action_mean, action_std = action[:, :self.dim_action], action[:, self.dim_action:]
        action_mean = torch.clamp(action_mean, -1, 1)
        action_std = torch.clamp(action_std, 0, 1)
        return action_mean, action_std

    def get_action(self, x):
        action_mean, action_std = self.forward(x)
        normal = Normal(action_mean, action_std)
        action = normal.rsample()
        return action


class Policy(nn.Module):
    def __init__(self, dim_hidden=256, activation=nn.LeakyReLU):
        super(Policy, self).__init__()

        self.dim_state_disc = 132
        self.dim_state_continuous = 23
        self.dim_state = self.dim_state_disc + self.dim_state_continuous
        self.dim_action = 6
        self.dim_hidden = dim_hidden

        # (s, a) -> s'
        self.EnvPolicy = EnvPolicy(self.dim_state_continuous, self.dim_state_disc, self.dim_action,
                                   activation=activation)

        # s -> a
        self.AgentPolicy = AgentPolicy(self.dim_state, self.dim_action, activation=activation)
        self.EnvPolicy.apply(init_weight)
        self.AgentPolicy.apply(init_weight)

        self.memory = Memory()

        to_device(self.EnvPolicy, self.AgentPolicy)

    def get_env_action(self, env_state, agent_action):
        return self.EnvPolicy.get_action(torch.cat([env_state, agent_action], dim=-1))

    def get_agent_action(self, env_state):
        action = self.AgentPolicy.get_action(env_state)
        return action

    def sample_state(self, expert_data):
        traj_idx = random.randint(0, expert_data.size(0) - 1)
        state = expert_data[traj_idx, :self.dim_state]
        return state.unsqueeze(0).to(device)

    def generate_batch(self, mini_batch_size, expert_data, traj_lens=10):
        """
        generate enough (state, action) pairs into memory, at least min_batch_size items.
        ######################
        :param mini_batch_size: number of pairs to generate
        :return: None
        """
        self.memory.clear()

        num_items = 0  # count generated (state, action) pairs

        env_state = self.sample_state(expert_data)  # sample init state from expert data

        while num_items < mini_batch_size:
            agent_action = self.get_agent_action(env_state)
            env_next_state = self.get_env_action(env_state, agent_action)
            mask = 1 if (num_items != 0 and num_items % traj_lens == 0) else 0

            state = torch.cat([env_state, agent_action], dim=-1).squeeze(0).detach().cpu().numpy()
            action = env_next_state.detach().cpu().numpy()

            self.memory.push(state, action, mask)

            num_items += 1

            if mask == 1:
                env_state = self.sample_state(expert_data)  # re-sample init state
            else:
                env_state = env_next_state

    def sample_batch(self, batch_size):
        """
        sample batch generate (state, action) pairs with mask.
        :param batch_size: mini_batch for update Discriminator
        :return: batch_gen, batch_mask
        """
        # sample batch (state, action) pairs from memory
        batch = self.memory.sample(batch_size)

        batch_state = FLOAT(np.stack(batch.state)).squeeze(1).to(device)
        batch_action = FLOAT(np.stack(batch.action)).squeeze(1).to(device)
        batch_mask = INT(np.stack(batch.mask)).to(device)

        assert batch_state.size(0) == batch_size, f"Expected batch size {batch_size} (s,a) pairs"

        return batch_state, batch_action, batch_mask

    def get_log_prob(self, state, action):
        # env policy
        disc = action[:, :self.dim_state_disc]
        continuous = action[:, self.dim_state_disc:]

        action_disc, action_continuous_mean, action_continuous_logstd = self.EnvPolicy(state)
        softmax_disc_action, _ = self.EnvPolicy.get_softmax_disc_action(action_disc)
        log_prob = FLOAT([]).to(device)
        # discrete log probability
        for i in range(n_actions):
            cur_m = OneHotCategorical(softmax_disc_action[:, action_slices[i]])
            cur_log_prob = cur_m.log_prob(disc[:, action_slices[i]]).view(-1, 1)
            log_prob = torch.cat((log_prob, cur_log_prob), dim=-1)
        # continuous log probability
        action_continuous_std = torch.exp(action_continuous_logstd)
        normal_dist = Normal(action_continuous_mean, action_continuous_std)
        continuous_log_prob = normal_dist.log_prob(continuous)
        # concat log probability
        log_prob = torch.cat([log_prob, continuous_log_prob], dim=-1)

        # agent policy
        agent_state = state[:, :self.dim_state]
        agent_action = state[:, self.dim_state:]
        mean, std = self.AgentPolicy(agent_state)
        normal = Normal(mean, std.exp())
        agent_log_prob = normal.log_prob(agent_action)

        log_prob = torch.cat([log_prob, agent_log_prob], dim=1)
        return log_prob
