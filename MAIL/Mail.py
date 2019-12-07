import random

from MAIL.MailPolicy import MailPolicy
from MAIL.RewardModel import RewardModel
from MAIL.ValueModel import ValueModel
from MAIL.ppo import GAE, PPO_step
from utils.utils import *


class MailModel:
    def __init__(self, expert_data, lr_d=0.0001, lr_g=0.0005, lr_v=0.0001, trajectory_num=300, batch_size=64,
                 ppo_epoch=16, mini_batch_size=64, epsilon=0.1, l2_reg=1e-4):
        self.expert_data = expert_data

        self.trajectory_num = trajectory_num
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.epsilon = epsilon
        self.l2_reg = l2_reg

        self.D = RewardModel()
        self.G = MailPolicy()
        self.V = ValueModel()

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)
        self.optim_V = optim.Adam(self.V.parameters(), lr=lr_v)

        self.loss_func = nn.BCELoss()

        to_device(self.G, self.D, self.V, self.loss_func)

    def train(self):
        writer = SummaryWriter()
        batch_num = (len(self.expert_data) + self.batch_size - 1) // self.batch_size

        for epoch in range(100):
            for i in range(batch_num):

                trajectory = self.G.generate_trajectory(self.trajectory_num)

                # sample gen trajectories
                batch_gen_old = random.sample(trajectory, self.batch_size)

                # sample expert trajectories
                idx = torch.randperm(len(self.expert_data))

                batch_expert_old = [self.expert_data[int(id)] for id in
                                    idx[i * self.batch_size:(i + 1) * self.batch_size]]

                # transform to 2D tensor
                batch_gen = FLOAT([]).to(device)
                batch_expert = FLOAT([])
                mask = INT([]).to(device)
                for g, e in zip(batch_gen_old, batch_expert_old):
                    batch_gen = torch.cat([batch_gen, g], dim=0)
                    batch_expert = torch.cat([batch_expert, e], dim=0)

                    temp_mask = torch.ones(g.shape[0], dtype=torch.int).to(device)
                    temp_mask[g.shape[0] - 1] = 0
                    mask = torch.cat([mask, temp_mask], dim=0)

                # gradient ascent update Reward
                for _ in range(1):
                    self.optim_D.zero_grad()

                    expert_o = self.D(batch_expert.to(device))
                    gen_o = self.D(batch_gen)

                    r_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device)) + \
                             self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    r_loss.backward()

                    self.optim_D.step()

                # PPO update joint policy
                with torch.no_grad():
                    gen_r = self.D(batch_gen)
                    value_o = self.V(batch_gen[:, :-1])
                    fixed_log_prob = self.G.get_log_prob(batch_gen[:, :-1], batch_gen[:, -1])

                advantages, returns = GAE(gen_r, mask, value_o, gamma=0.95, lam=0.95)

                for _ in range(5):
                    new_index = torch.randperm(batch_gen.shape[0]).to(device)

                    batch_gen_state, batch_gen_action, fixed_log_prob, returns, advantages = \
                        batch_gen[:, :-1][new_index].clone(), batch_gen[:, -1][new_index].clone(), \
                        fixed_log_prob[new_index].clone(), returns[new_index].clone(), advantages[new_index].clone()

                    # for j in range(self.ppo_epoch):
                    #     ind = slice(j * self.mini_batch_size, min((j + 1) * self.mini_batch_size, batch_gen.shape[0]))
                    #     gen_state_mini, gen_action_mini, fixed_log_prob_mini, returns_mini, advantages_mini = \
                    #         batch_gen_state[ind], batch_gen_action[ind], fixed_log_prob[ind], returns[ind], advantages[
                    #             ind]

                    PPO_step(self.G, self.V, self.optim_G, self.optim_V, batch_gen_state, batch_gen_action,
                                 returns, advantages, fixed_log_prob, self.epsilon, self.l2_reg)

                writer.add_scalars('MAIL/train_loss', {'Batch_R_loss': r_loss,
                                                       'Batch_reward': gen_r.mean()},
                                   epoch * batch_num + i)
                print(f'Epoch: {epoch}, Batch: {i}, Batch loss: {r_loss.cpu().detach().numpy():.4f}, '
                      f'Batch reward: {gen_r.mean().cpu().detach().numpy():.4f}')

            # if (epoch + 1) % 1 == 0:
            self.save_model()

    def save_model(self):
        torch.save(self.G.state_dict(), r'../model/policy.pt')
        torch.save(self.V.state_dict(), r'../model/value.pt')
        torch.save(self.D.state_dict(), r'../model/discriminator.pt')
