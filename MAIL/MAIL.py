import random

from GAN_SD.GeneratorModel import GeneratorModel
from MAIL.EnginePolicy import EnginePolicy
from MAIL.RewardModel import RewardModel
from MAIL.UserLeaveModel import UserLeaveModel
from MAIL.UserPolicy import UserPolicy
from utils.utils import *


class MailModel:
    def __init__(self, expert_data, lr_r=0.0005, trajectory_num=10000, batch_size=32):
        self.expert_data = expert_data

        self.trajectory_num = trajectory_num
        self.batch_size = batch_size

        self.UserModel = GeneratorModel()
        self.UserModel.load()

        self.UserPolicy = UserPolicy()
        self.UserLeaveModel = UserLeaveModel()
        self.RewardModel = RewardModel()

        self.EnginePolicy = EnginePolicy()

        self.optim_R = optim.Adam(self.EnginePolicy.parameters(), lr=lr_r)
        self.loss_func = nn.BCELoss()

    def train(self):
        batch_num = (len(self.expert_data) + self.batch_size - 1) // self.batch_size
        for epoch in range(100):

            for i in range(batch_num):

                # sample J trajectories
                trajectory = []
                for j in range(self.trajectory_num):
                    tao_j = []
                    # sample user from GAN-SD distribution
                    s, _ = self.UserModel.generate()
                    # get user's leave page index from leave model
                    leave_page_index = self.UserLeaveModel(s)
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
                    self.optim_R.zero_grad()

                    expert_o = self.RewardModel(batch_expert.to(device))
                    gen_o = self.RewardModel(batch_gen)

                    r_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device)) + \
                             self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    r_loss.backward()

                    self.optim_R.step()

                # TODO PPO update joint policy

    def save_model(self):
        torch.save(self.UserPolicy.state_dict(), './model/UserPolicy.pt')
        torch.save(self.EnginePolicy.state_dict(), './model/EnginePolicy.pt')
        torch.save(self.UserLeaveModel.state_dict(), './model/UserLeaveModel.pt')
        torch.save(self.RewardModel.state_dict(), '../model/RewardModel.pt')
