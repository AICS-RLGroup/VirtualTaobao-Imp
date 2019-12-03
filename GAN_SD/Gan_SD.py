#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2019/12/2 下午7:14
from GAN_SD.DiscriminatorModel import DiscriminatorModel
from GAN_SD.GeneratorModel import GeneratorModel
from utils.utils import *


class GanSDModel:
    def __init__(self, dim_user, dim_seed, lr_g, lr_d, expert_users, batch_size=256, alpha=1.0, beta=1.0):
        self.dim_user = dim_user
        self.dim_seed = dim_seed

        self.expert_users = expert_users[:, :88]
        self.batch_size = batch_size
        self.n_expert_users = self.expert_users.size(0)

        self.alpha = alpha
        self.beta = beta

        self.G = GeneratorModel(activation=nn.Tanh)
        self.D = DiscriminatorModel(activation=nn.LeakyReLU)

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)
        self.loss_func = nn.BCELoss()

        to_device(self.G, self.D, self.loss_func)

    @log_func("Train GAN-SD Model")
    def train(self):
        n_batch = (self.n_expert_users + self.batch_size - 1) // self.batch_size

        time_start = time.time()
        writer = SummaryWriter()

        for epoch in range(1000):
            idx = torch.randperm(self.n_expert_users)
            for i in range(n_batch):

                # sample minibatch from expert users
                batch_expert = self.expert_users[idx[i * self.batch_size:(i + 1) * self.batch_size]]
                # sample minibatch from generator
                batch_seed = torch.normal(torch.zeros(batch_expert.size(0), self.dim_seed),
                                          torch.ones(batch_expert.size(0), self.dim_seed)).to(device)
                batch_gen, _ = self.generate(batch_seed)

                # gradient ascent update discriminator
                for _ in range(1):
                    self.optim_D.zero_grad()

                    expert_o = self.D(batch_expert.to(device))
                    gen_o = self.D(batch_gen.detach())

                    d_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device)) + \
                             self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    d_loss.backward()

                    self.optim_D.step()

                # gradient ascent update generator
                for _ in range(3):
                    self.optim_G.zero_grad()
                    # sample minibatch from generator
                    batch_seed = torch.normal(torch.zeros(batch_expert.size(0), self.dim_seed),
                                              torch.ones(batch_expert.size(0), self.dim_seed)).to(device)
                    batch_gen, batch_gen_feature = self.generate(batch_seed)
                    gen_o = self.D(batch_gen.detach())

                    kl = self.get_kl(batch_gen_feature, batch_expert)
                    # g_loss = -(gen_o.sum() +
                    #          self.alpha * self.get_prob_entropy(batch_gen)[1] -
                    #          self.beta * kl)
                    g_loss = self.loss_func(gen_o, torch.ones_like(gen_o, device=device)) + self.beta * kl - \
                             self.alpha * self.get_prob_entropy(batch_gen)[1]
                    g_loss.backward()
                    self.optim_G.step()

                writer.add_scalars('GAN_SD/train_loss', {'discriminator_GAN_SD': d_loss,
                                                         'generator_GAN_SD': g_loss,
                                                         'KL_divergence': kl},
                                   epoch * n_batch + i)

                if i % 10 == 0:
                    cur_time = time.time() - time_start
                    eta = cur_time / (i + 1) * (n_batch - i - 1)
                    print('Epoch %2d Batch %4d G_Loss %.3f KL %.3f D_Loss %.3f. Time elapsed: %.2fs ETA : %.2fs' % (
                        epoch, i, g_loss.cpu().detach().numpy(), kl.cpu().detach().numpy(),
                        d_loss.cpu().detach().numpy(), cur_time, eta))

    # generate random user with one-hot encoded feature
    def generate(self, z=None):
        if z is None:
            z = torch.rand((1, self.dim_seed)).to(device)  # generate 1 random seed
        x = self.get_prob_entropy(self.G(z))[0]  # softmax_feature
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
        one_hot = FLOAT([]).to(device)
        for i in range(11):
            tmp = torch.zeros_like(features[i], device=device)
            one_hot = torch.cat((one_hot, tmp.scatter_(1, torch.multinomial(features[i], 1), 1)),
                                dim=-1)  # 根据softmax feature 生成one-hot的feature
        return one_hot, features

    def get_prob_entropy(self, x):
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
        entropy = 0.0
        softmax_feature = FLOAT([]).to(device)
        for i in range(11):
            softmax_feature = torch.cat([softmax_feature, F.softmax(features[i], dim=1)], dim=-1)
            entropy += -(F.log_softmax(features[i], dim=1) * F.softmax(features[i], dim=1)).sum(dim=1).mean()
        return softmax_feature, entropy

    def get_kl(self, batch_gen_feature, batch_expert):
        distributions = [None] * 11
        distributions[0] = batch_expert[:, :8]
        distributions[1] = batch_expert[:, 8:16]
        distributions[2] = batch_expert[:, 16:27]
        distributions[3] = batch_expert[:, 27:38]
        distributions[4] = batch_expert[:, 38:49]
        distributions[5] = batch_expert[:, 49:60]
        distributions[6] = batch_expert[:, 60:62]
        distributions[7] = batch_expert[:, 62:64]
        distributions[8] = batch_expert[:, 64:67]
        distributions[9] = batch_expert[:, 67:85]
        distributions[10] = batch_expert[:, 85:88]

        kl = 0.0
        for i in range(11):
            kl += (F.softmax(batch_gen_feature[i].to(device), dim=1) *
                   (F.log_softmax(batch_gen_feature[i].to(device), dim=1) -
                    F.log_softmax(distributions[i].to(device), dim=1))).sum(dim=1).mean()

        return kl

    def save_model(self):
        torch.save(self.G.state_dict(), r'./model/user_G.pt')
        torch.save(self.D.state_dict(), r'./model/user_D.pt')
