from utils.utils import *


class UserLeaveModel(nn.Module):
    def __init__(self, n_input=88, n_output=101):
        super(UserLeaveModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )
        self.model.apply(init_weight)

    def forward(self, user):
        x = self.model(user)
        leave_page_index = torch.multinomial(F.softmax(x, dim=1), 1)
        return leave_page_index
