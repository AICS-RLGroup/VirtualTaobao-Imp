from utils.utils import *


class OUNoise:
    def __init__(self, dim_state_action=88 + 27 + 1 + 1, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.dim_state_action = dim_state_action
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.dim_state_action) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim_state_action) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class DiscriminatorModel(nn.Module):
    def __init__(self, n_input=88 + 27 + 1 + 1, n_hidden=256, n_output=1, activation=nn.LeakyReLU):
        super(DiscriminatorModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_output),
            nn.Dropout(p=0.6),
            nn.Sigmoid()
        )

        self.Noise = OUNoise(n_input)
        self.model.apply(init_weight)

    def forward(self, x):
        noise = torch.zeros_like(x)
        for i in range(noise.size(0)):
            noise[i] += FLOAT(self.Noise.noise()).to(device)
        x += noise
        return self.model(x)
