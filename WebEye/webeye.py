import math

from algorithms.ppo import GAE, PPO_step
from WebEye.discriminator import Discriminator
from WebEye.policy import Policy
from WebEye.value import Value
from utils.utils import *


class WebEyeModel:
    def __init__(self, expert_data, lr_d=1e-4, lr_g=3e-4, lr_v=1e-3, epochs=200000, batch_size=64, ppo_epoch=10,
                 ppo_minibatch_size=8, epsilon=0.2, l2_reg=1e-4):
        self.expert_data = expert_data

        self.epochs = epochs
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.ppo_minibatch_size = ppo_minibatch_size
        self.epsilon = epsilon
        self.l2_reg = l2_reg

        self.D = Discriminator()
        self.G = Policy()
        self.V = Value()

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)
        self.optim_V = optim.Adam(self.V.parameters(), lr=lr_v)

        self.loss_func = nn.BCELoss()

        to_device(self.G, self.D, self.V, self.loss_func)

    def train(self):
        writer = SummaryWriter()
        batch_num = (len(self.expert_data) + self.batch_size - 1) // self.batch_size

        for epoch in range(self.epochs):
            # shuffle expert data
            idx = torch.randperm(len(self.expert_data))
            # generate (state, action) pairs
            expert_state_action_num = 1024
            self.G.generate_batch(expert_state_action_num, self.expert_data)
            # sample generated (state, action) pairs from memory
            batch_gen_state, batch_gen_action, mask = self.G.sample_batch(self.batch_size)

            ############################
            # update Discriminator
            ############################
            for i in range(batch_num):
                # sample a batch of expert trajectories
                batch_expert = self.expert_data[idx[i * self.batch_size:(i + 1) * self.batch_size], :]

                # update Discriminator adaptively
                update_frequency = 1 if epoch < 40 else 20

                if i % update_frequency == 0:
                    self.optim_D.zero_grad()

                    expert_o = self.D(batch_expert.to(device))
                    gen_o = self.D(torch.cat([batch_gen_state, batch_gen_action], dim=1).to(device))

                    e_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device))
                    g_loss = self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    d_loss = g_loss + e_loss

                    # g_loss = self.loss_func(gen_o, torch.ones_like(gen_o, device=device))
                    # e_loss = self.loss_func(expert_o, torch.zeros_like(expert_o, device=device))
                    # d_loss = -(g_loss + e_loss)

                    d_loss.backward()

                    self.optim_D.step()

                writer.add_scalars('WebEye/Discriminator',
                                   {'Batch_D_loss': d_loss,
                                    'Batch_G_loss': g_loss,
                                    'Batch_E_loss': e_loss
                                    },
                                   epoch * batch_num + i)

                writer.add_scalars('WebEye/Reward',
                                   {'Batch_G_reward': gen_o.mean(),
                                    'Batch_E_reward': expert_o.mean()
                                    },
                                   epoch * batch_num + i)
                print("=" * 100 + "Discriminator")

                print(f'Epoch: {epoch}, Batch: {i}, Batch E loss: {e_loss.detach().cpu().numpy():.4f}, '
                      f'Batch G loss: {g_loss.cpu().detach().numpy(): .4f}, '
                      f'Batch D loss: {d_loss.cpu().detach().numpy(): .4f}, '
                      f'Batch G reward: {gen_o.mean().cpu().detach().numpy():.4f}, '
                      f'Batch E reward: {expert_o.mean().detach().cpu().numpy(): .4f}')

            with torch.no_grad():
                gen_r = self.D(torch.cat([batch_gen_state, batch_gen_action.type(torch.float)], dim=1).to(device))
                value_o = self.V(batch_gen_state)
                fixed_log_prob = self.G.get_log_prob(batch_gen_state, batch_gen_action)

            advantages, returns = GAE(-torch.log(1 - gen_r + 1e-6), mask, value_o, gamma=0.99, lam=0.96)

            ##############################
            # update Generator using PPO
            ##############################
            """perform mini-batch PPO update"""
            optim_iter_num = int(math.ceil(batch_gen_state.shape[0] / self.ppo_minibatch_size))
            for k in range(self.ppo_epoch):
                new_index = torch.randperm(batch_gen_state.shape[0]).to(device)

                mini_batch_gen_state, mini_batch_gen_action, mini_batch_fixed_log_prob, mini_batch_returns, \
                mini_batch_advantages = batch_gen_state[new_index], batch_gen_action[new_index], fixed_log_prob[
                    new_index], returns[new_index], advantages[new_index]

                for j in range(optim_iter_num):
                    ind = slice(j * self.ppo_minibatch_size,
                                min((j + 1) * self.ppo_minibatch_size, batch_gen_state.shape[0]))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                        mini_batch_gen_state[ind], mini_batch_gen_action[ind], mini_batch_advantages[ind], \
                        mini_batch_returns[ind], mini_batch_fixed_log_prob[ind]

                    v_loss, p_loss = PPO_step(self.G, self.V, self.optim_G, self.optim_V, states_b,
                                              actions_b,
                                              returns_b, advantages_b, fixed_log_probs_b,
                                              self.epsilon, self.l2_reg)

                writer.add_scalars('WebEye/Generator',
                                   {'Batch_V_loss': v_loss,
                                    'Batch_P_loss': p_loss
                                    },
                                   epoch * self.ppo_epoch + k)

                print("=" * 100 + "Generator")
                print(f'Epoch: {epoch}, Batch V loss: {v_loss.detach().cpu().numpy():.4f}, '
                      f'Batch P loss: {p_loss.detach().cpu().numpy(): .10f}'
                      )

            self.save_model()

            torch.cuda.empty_cache()

    def save_model(self):
        torch.save(self.G.state_dict(), r'../model/web_policy.pt')
        torch.save(self.V.state_dict(), r'../model/web_value.pt')
        torch.save(self.D.state_dict(), r'../model/web_discriminator.pt')
