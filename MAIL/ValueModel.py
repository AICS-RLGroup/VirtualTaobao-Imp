from utils.utils import *
import torch
import torch.nn as nn


class ValueModel(nn.Module):
    def __init__(self, user_state_dim = 88+27+1, hidden_layer_size=(256), output_dim=1, activiation=nn.LeakyReLU):
        super().__init__()
        self.models = nn.Sequential(
            nn.Linear(user_state_dim, hidden_layer_size[0]),
            activiation(),
            nn.Linear(hidden_layer_size[0],1)
        )

        self.models.apply(init_weight)

    def forward(self, x):
        value = self.model(x)

        return value

