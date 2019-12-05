import random

from MAIL.MailPolicy import MailPolicy
from MAIL.RewardModel import RewardModel
from utils.utils import *


class MailModel:
    def __init__(self, expert_data, lr_d=0.0001, lr_g=0.0005, trajectory_num=10000, batch_size=32):
        self.expert_data = expert_data

        self.trajectory_num = trajectory_num
        self.batch_size = batch_size

        self.D = RewardModel()
        self.G = MailPolicy()

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)

        self.loss_func = nn.BCELoss()

    def train(self):
        batch_num = (len(self.expert_data) + self.batch_size - 1) // self.batch_size
        for epoch in range(100):

            for i in range(batch_num):

                trajectory = self.G.generate(self.trajectory_num)

                # sample gen trajectories
                batch_gen_old = random.sample(trajectory, self.batch_size)

                # sample expert trajectories
                idx = torch.randperm(len(self.expert_data))

                batch_expert_old = self.expert_data[idx[i * self.batch_size:(i + 1) * self.batch_size]]

                # transform to 2D tensor
                batch_gen = FLOAT([]).to(device)
                batch_expert = FLOAT([])
                for g, e in zip(batch_gen_old, batch_expert_old):
                    batch_gen = torch.cat([batch_gen, g], dim=0)
                    batch_expert = torch.cat([batch_expert, e], dim=0)

                # gradient ascent update Reward
                for _ in range(1):
                    self.optim_D.zero_grad()

                    expert_o = self.D(batch_expert.to(device))
                    gen_o = self.D(batch_gen)

                    r_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device)) + \
                             self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    r_loss.backward()

                    self.optim_D.step()

                # TODO PPO update joint policy

    def save_model(self):
        torch.save(self.G.state_dict(), r'./model/policy.pt')
        torch.save(self.V.state_dict(), r'./model/value.pt')
        torch.save(self.D.state_dict(), r'./model/discriminator.pt')
