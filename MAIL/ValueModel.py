from utils.utils import *


class ValueModel(nn.Module):
    def __init__(self, dim_user_state=88 + 27 + 1, dim_hidden=256, dim_out=1, activation=nn.LeakyReLU):
        super().__init__()
        self.models = nn.Sequential(
            nn.Linear(dim_user_state, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_out)
        )

        self.models.apply(init_weight)

    def forward(self, x):
        value = self.model(x)
        return value
