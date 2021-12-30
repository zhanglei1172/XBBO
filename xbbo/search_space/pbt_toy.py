import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from xbbo.utils.constants import MAXINT

from xbbo.core import TestFunction

class LossIsNaN(Exception):
    pass




class Model(TestFunction):

    def __init__(self, cfg, seed, **kwargs):
        # np.random.seed(cfg.GENERAL.random_seed)
        self.cfg = cfg
        # self.dim = 30
        # assert self.dim % 2 == 0
        super().__init__(seed=seed)

        self.api_config = self._load_api_config()
        torch.seed(self.rng.randint(MAXINT))
        torch.manual_seed(self.rng.randint(MAXINT))
        self.device = torch.device(kwargs.get('device', 'cpu'))

        self.theta = Parameter(torch.FloatTensor([0.9, 0.9]).to(self.device))
        # self.opt_wrap = lambda params: optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.opt = optim.SGD([self.theta], lr=0.01)
        self.step_num = 0
        self.history_hp = [] # for record strategy
        self.trajectory_hp = []
        self.trajectory_loss = [] # 记录该个体score过程
        self.history_loss = [] # 记录使用了（考虑权重迁移）hp-stategy后的score过程
        self.hp = torch.empty(2, device=self.device)
        self.obj_val_func = lambda theta: 1.2 - (theta ** 2).sum()
        self.obj_train_func = lambda theta, h: 1.2 - ((h * theta) ** 2).sum()

        self.trajectory_theta = []

    def __len__(self): # one epoch has how many batchs
        return 1

    def update_hp(self, params: dict):
        self.history_hp.append((self.step_num, params)) # 在该steps上更改超参，acc为该step时的结果（受该step*前*所有超参影响）
        self.trajectory_hp.append((self.step_num, params))
        self.trajectory_theta.append(self.theta.detach().cpu().numpy())
        self.hp[0] = params['h1']
        self.hp[1] = params['h2']

    def step(self, num): # train need training(optimizer)
        for it in range(num):
            self.trajectory_theta.append(self.theta.detach().cpu().numpy())
            loss = self.obj_train_func(self.theta, self.hp)
            if np.isnan(loss.item()):
                print("Loss is NaN.")
                self.step_num += 1
                return
                # raise LossIsNaN
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.step_num += 1


    def evaluate(self): # val no training need(optimizer)
        with torch.no_grad():
            loss = self.obj_val_func(self.theta).item()
        self.loss = np.inf if np.isnan(loss) else loss
        self.trajectory_loss.append((self.step_num, self.loss))
        self.history_loss.append((self.step_num, self.loss))
        return self.loss

    def load_checkpoint(self, checkpoint):
        with torch.no_grad():
            self.theta.set_(checkpoint['model_state_dict'])
        # self.opt.load_state_dict(checkpoint['optim_state_dict'])

    def save_checkpoint(self):
        checkpoint = dict(model_state_dict=self.theta.data.clone())
        return checkpoint

    def _load_api_config(self):
        return {
            'h1': {
                'type': 'float', 'warp': 'linear', 'range': [0, 1]},
            'h2': {
                'type': 'float', 'warp': 'linear', 'range': [0, 1]
            }
        }

