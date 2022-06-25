import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from xbbo.core.constants import MAXINT

from xbbo.search_algorithm.pbt_optimizer import Abstract_PBT_Model


class LossIsNaN(Exception):
    pass


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5,5))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MnistShuffleTrainData(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(0)
        self.idx_shuffle = np.random.permutation(len(self))

    def __getitem__(self, item):
        return super().__getitem__(self.idx_shuffle[item])

class Model(Abstract_PBT_Model):
    trn_dataset = None
    tst_dataset = None
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed, **kwargs)
        # np.random.seed(cfg.GENERAL.random_seed)
        # self.dim = 30
        # assert self.dim % 2 == 0

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,))])
        if Model.trn_dataset is None:
            data_path = kwargs.get('data_path', './data/')
            Model.trn_dataset = MnistShuffleTrainData(data_path, train=True, download=True,
                                            transform=transform)
            Model.tst_dataset = datasets.MNIST(data_path, train=False, download=True,
                                                transform=transform)
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.train_loader = DataLoader(Model.trn_dataset, batch_size=64, shuffle=False,num_workers=0)
        self.test_loader = DataLoader(Model.tst_dataset, batch_size=64, shuffle=False, num_workers=0)
        self.iter_train_loader = iter(self.train_loader)
        self.net = ConvNet().to(self.device)
        # self.opt_wrap = lambda params: optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.opt = optim.SGD(self.net.parameters(), lr=0.11, momentum=0.9)
        self.loss_fn = F.nll_loss
        self.step_num = 0
        self.ready = False # not ready
        self.history_hp = [] # for record strategy
        self.trajectory_hp = []
        self.trajectory_loss = [] # 记录该个体loss过程
        self.history_loss = [] # 记录使用了（考虑权重迁移）hp-stategy后的loss过程

    def __len__(self): # one epoch has how many batchs
        return len(self.train_loader)

    def update_hp(self, params: dict):
        self.history_hp.append((self.step_num, params)) # 在该steps上更改超参，acc为该step时的结果（受该step*前*所有超参影响）
        self.trajectory_hp.append((self.step_num, params))
        for hyperparam_name, v in params.items():
            for param_group in self.opt.param_groups:
                param_group[hyperparam_name] = v

    def _one_step(self, **kwargs) -> float:
        try:
            inp, target = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            inp, target = next(self.iter_train_loader)
        inp = inp.to(self.device)
        target = target.to(self.device)
        output = self.net(inp)
        loss = self.loss_fn(output, target)
        if np.isnan(loss.item()):
            print("Loss is NaN.")
            return np.inf
        # raise LossIsNaN
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def step(self, num, **kwargs): # train need training(optimizer)
        self.net.train()
        for it in range(num):
            loss: float = self._one_step(**kwargs)
            if not np.isfinite(loss):
                self.step_num += num - it
                return
            self.step_num += 1
        

    def evaluate(self): # val no training need(optimizer)
        correct = 0
        self.net.eval()
        with torch.no_grad():
            for inp, target in self.test_loader:
                inp = inp.to(self.device)
                target = target.to(self.device)
                output = self.net(inp)
                correct += (output.max(1)[1] == target).sum().cpu().item()
        acc = 100 * correct / len(self.tst_dataset)
        self.loss = np.inf if np.isnan(acc) else -acc
        self.trajectory_loss.append((self.step_num, self.loss))
        self.history_loss.append((self.step_num, self.loss))
        return self.loss

    def load_checkpoint(self, checkpoint):
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optim_state_dict'])

    def save_checkpoint(self):
        '''
        Optional Serialization to disk
        '''
        checkpoint = dict(model_state_dict=self.net.state_dict(),
                          optim_state_dict=self.opt.state_dict())
        return checkpoint

    @classmethod
    def get_configuration_space(cls, seed):
        if hasattr(cls, "cs"):
            return cls.configuration_space
        cls.configuration_space = ConfigurationSpace(seed=seed)
        cls.configuration_space.add_hyperparameter(UniformFloatHyperparameter('lr', 1e-5, 1, log=True))
        cls.configuration_space.add_hyperparameter(UniformFloatHyperparameter('momentum', 0.1, 0.999))
        return cls.configuration_space
