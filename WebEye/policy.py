#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午4:17
import random

from torch.distributions import MultivariateNormal

from custom.MultiOneHotCategorical import MultiOneHotCategorical
from custom.MultiSoftMax import MultiSoftMax
from utils.replay_memory import Memory
from utils.utils import *

# 离散 action 对应
action_sizes = [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]


# (s, a) -> s'
class EnvPolicy(nn.Module):
    def __init__(self, dim_state_continuous=23, dim_state_disc=132, dim_action=6, dim_hidden=256,
                 activation=nn.LeakyReLU, log_std=0.0):
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

        # predict discrete action using custom softmax
        self.action_disc = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_state_disc),
            MultiSoftMax(0, self.dim_state_disc, action_sizes))

        # predict mean and std for continuous action
        self.action_continuous_mean = nn.Linear(self.dim_hidden, self.dim_state_continuous)
        self.action_continuous_log_std = nn.Parameter(torch.ones(1, self.dim_state_continuous) * log_std)

    def forward(self, x):
        """
        :param x: (state, action)
        :return: next_state
        """
        x = self.common(x)
        action_disc = self.action_disc(x)
        action_continuous_mean = self.action_continuous(x)
        action_continuous_log_std = self.action_continuous_log_std.expand_as(action_continuous_mean)
        return action_disc, action_continuous_mean, action_continuous_log_std

    def get_onehot_disc_action(self, x):
        """
        get softmax discrete action
        :param x: multi categorical probs
        :return:
        """
        # 15个one-hot向量拼接而成，大小分别是 [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]
        m = MultiOneHotCategorical(x, action_sizes)
        return m.sample()

    def get_action(self, x):
        action_disc_probs, action_continuous_mean, action_continuous_log_std = self.forward(x)
        action_disc = self.get_onehot_disc_action(action_disc_probs)

        action_continuous_std = action_continuous_log_std.exp()
        multi_normal_dist = MultivariateNormal(action_continuous_mean, torch.diag_embed(action_continuous_std))
        action_continuous = multi_normal_dist.rsample()

        action = torch.cat([action_disc, action_continuous], dim=-1)
        return action


# s -> a
class AgentPolicy(nn.Module):
    def __init__(self, dim_state=155, dim_action=6, dim_hidden=128,
                 activation=nn.LeakyReLU, log_std=0.0):
        super(AgentPolicy, self).__init__()

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden

        self.action = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_action)
        )

        self.action_log_std = nn.Parameter(torch.ones(1, self.dim_action) * log_std)

    def forward(self, x):
        """
        :param x: state
        :return: action
        """
        action_mean = self.action(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        return action_mean, action_log_std

    def get_action(self, x):
        action_mean, action_log_std = self.forward(x)
        action_std = action_log_std.exp()

        multi_normal = MultivariateNormal(action_mean, torch.diag_embed(action_std))
        action = multi_normal.rsample()
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

    def generate_batch(self, mini_batch_size, expert_data, traj_length=10):
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
            mask = 1 if (num_items != 0 and num_items % traj_length == 0) else 0

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

        action_disc_probs, action_continuous_mean, action_continuous_log_std = self.EnvPolicy(state)
        action_disc_log_prob = MultiOneHotCategorical(action_disc_probs, action_sizes).log_prob(disc)

        action_continuous_std = torch.exp(action_continuous_log_std)
        normal_dist = MultivariateNormal(action_continuous_mean, torch.diag_embed(action_continuous_std))
        action_continuous_log_prob = normal_dist.log_prob(continuous)

        # agent policy
        agent_state = state[:, :self.dim_state]
        agent_action = state[:, self.dim_state:]
        mean, log_std = self.AgentPolicy(agent_state)
        std = log_std.exp()
        normal = MultivariateNormal(mean, torch.diag_embed(std))
        agent_log_prob = normal.log_prob(agent_action)

        print(agent_log_prob.shape, action_disc_log_prob.shape, action_continuous_log_prob.shape)
        log_prob = agent_log_prob + action_disc_log_prob + action_continuous_log_prob
        return log_prob
