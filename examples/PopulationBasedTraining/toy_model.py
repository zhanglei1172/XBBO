
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.optim import SGD
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from xbbo.core.constants import MAXINT
from xbbo.search_algorithm.pbt_optimizer import Abstract_PBT_Model

class Model(Abstract_PBT_Model):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)
        # self.api_config = self._load_api_config()
        torch.manual_seed(self.rng.randint(MAXINT))
        self.device = torch.device(kwargs.get('device', 'cpu'))

        self.theta = Parameter(torch.FloatTensor([0.9, 0.9]).to(self.device))
        # self.opt_wrap = lambda params: optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.opt = SGD([self.theta], lr=0.01)
        self.history_hp = []  # for record strategy
        self.trajectory_hp = []
        self.trajectory_loss = []  # 记录该个体score过程
        self.history_loss = []  # 记录使用了（考虑权重迁移）hp-stategy后的score过程
        self.hp = torch.empty(2, device=self.device)
        self.obj_val_func = lambda theta: 1.2 - (theta**2).sum()
        self.obj_train_func = lambda theta, h: 1.2 - ((h * theta)**2).sum()

        self.trajectory_theta = []

    def __len__(self):  # one epoch has how many batchs
        return 1

    def update_hp(self, params: dict):
        self.history_hp.append(
            (self.step_num,
             params))  # 在该steps上更改超参，acc为该step时的结果（受该step*前*所有超参影响）
        self.trajectory_hp.append((self.step_num, params))
        self.trajectory_theta.append(self.theta.detach().cpu().numpy())
        self.hp[0] = params['h1']
        self.hp[1] = params['h2']

    def _one_step(self, **kwargs):  # train need training(optimizer)
        self.trajectory_theta.append(self.theta.detach().cpu().numpy())
        loss = -self.obj_train_func(self.theta, self.hp)
        if np.isnan(loss.item()):
            print("Loss is NaN.")
            return
            # raise LossIsNaN
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def evaluate(self):  # val no training need(optimizer)
        with torch.no_grad():
            loss = -self.obj_val_func(self.theta).item()
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
    
    @classmethod
    def get_configuration_space(cls, seed):
        if hasattr(cls, "cs"):
            return cls.configuration_space
        cls.configuration_space = ConfigurationSpace(seed=seed)
        x1 = UniformFloatHyperparameter("h1", 0, 1, default_value=0)
        x2 = UniformFloatHyperparameter("h2", 0, 1, default_value=0)
        cls.configuration_space.add_hyperparameters([x1, x2])
        return cls.configuration_space
