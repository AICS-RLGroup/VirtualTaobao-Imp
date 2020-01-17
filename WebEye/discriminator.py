from utils.OU_noise import OUNoise
from utils.utils import *


class Discriminator(nn.Module):
    def __init__(self, n_input=155 + 6 + 155, n_hidden=128, n_output=1, activation=nn.LeakyReLU):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_output),
            nn.Sigmoid()
        )

        self.model.apply(init_weight)

    def forward(self, x):
        noise = OUNoise(x.size())
        x += noise.sample()
        return self.model(x)
