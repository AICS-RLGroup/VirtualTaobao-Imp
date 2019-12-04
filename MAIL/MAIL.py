from MAIL.ActionModel import ActionModel
from MAIL.LeaveModel import LeaveModel
from MAIL.RewardModel import RewardModel
from MAIL.EngineModel import EngineModel
from GAN_SD.Gan_SD import GanSDModel
from utils.utils import *

class MailModel():
    def __init__(self,expert_data,user_distribution,trajectory_num,batch_size=256):
        self.expert_data = expert_data
        self.user = user_distribution

        self.trajectory_num = trajectory_num
        self.batch_size = batch_size

        self.A = ActionModel()
        self.L = LeaveModel()
        self.R = RewardModel()
        self.E = EngineModel()
        self.G = GanSDModel()

        self.loss_func = nn.BCELoss()

    def train(self):
        batch_num = (len(self.expert_data) + self.batch_size - 1) // self.batch_size
        for epoch in range(100):
            trajectory = []
            for j in range(self.trajectory_num):
                tao_j = []
                s,_ = self.G.generate()
                a = self.E(s)
                s_c = torch.cat((s,a),dim=1)
                while True:

                    pass

            idx = torch.randperm(len(self.expert_data))
            for i in range(batch_num):
                batch_expert = self.expert_data[idx[i * self.batch_size:(i + 1) * self.batch_size]]


    def save_model(self):
        torch.save(self.A.state_dict(),'../model/ActionModel.pt')
        torch.save(self.L.state_dict(),'../model/LeaveModel.pt')
        torch.save(self.R.state_dict(),'../model/RewardModel')