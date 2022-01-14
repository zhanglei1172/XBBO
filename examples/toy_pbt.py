import torch
from torch.nn.parameter import Parameter
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.search_algorithm.pbt_optimizer import PBT

from xbbo.utils.constants import MAXINT
from xbbo.core.abstract_model import TestFunction


class Model(TestFunction):
    def __init__(self, seed, **kwargs):
        super().__init__(seed=seed)

        self.api_config = self._load_api_config()
        torch.manual_seed(self.rng.randint(MAXINT))
        self.device = torch.device(kwargs.get('device', 'cpu'))

        self.theta = Parameter(torch.FloatTensor([0.9, 0.9]).to(self.device))
        # self.opt_wrap = lambda params: optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.opt = SGD([self.theta], lr=0.01)
        self.step_num = 0
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

    def step(self, num):  # train need training(optimizer)
        for it in range(num):
            self.trajectory_theta.append(self.theta.detach().cpu().numpy())
            loss = -self.obj_train_func(self.theta, self.hp)
            if np.isnan(loss.item()):
                print("Loss is NaN.")
                self.step_num += 1
                return
                # raise LossIsNaN
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.step_num += 1

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


if __name__ == "__main__":
    epoch_num = 100
    rng = np.random.RandomState(42)

    cs = DenseConfigurationSpace(seed=rng.randint(MAXINT))
    x1 = UniformFloatHyperparameter("h1", 0, 1, default_value=0)
    x2 = UniformFloatHyperparameter("h2", 0, 1, default_value=0)
    cs.add_hyperparameters([x1, x2])
    # define black box optimizer
    pbt = PBT(space=cs, pop_size=2, seed=rng.randint(MAXINT))

    population_model = [
        Model(seed=rng.randint(MAXINT)) for _ in range(pbt.pop_size)
    ]
    finished = False
    interval = 1
    pbt.init_model_hp(population_model)

    for i in range(int(epoch_num*len(population_model[0]))):
        while not finished:
            for i in range(pbt.pop_size):
                population_model[i].step(
                    int(interval * len(population_model[i])))
                if population_model[i].step_num == int(
                        len(population_model[i]) * epoch_num):
                    finished = True
            for i in range(pbt.pop_size):
                population_model[i].evaluate()
            losses = [net.loss for net in population_model]
            if finished:
                break
            pbt.exploit_and_explore_toy(population_model, losses)


    best_individual_index = np.argmin(losses)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(pbt.pop_size):
        desc_data = np.array(population_model[i].history_loss)
        desc_data[:, 0] /= len(population_model[-1])
        ax1.plot(desc_data[:, 0], desc_data[:, 1], alpha=0.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    for i in range(pbt.pop_size):
        desc_data = np.array(population_model[i].trajectory_theta)
        # desc_data[:, 0] /= self.interval * len(self.population_model[-1])
        ax2.scatter(desc_data[:, 0], desc_data[:, 1], s=2, alpha=0.5)
    # ax2.axis('equal')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"$\theta_1$")
    ax2.set_ylabel(r"$\theta_2$")
    plt.suptitle("PBT toy example")
    plt.tight_layout()
    plt.savefig('./out/PBT_toy_.png')
    plt.show()

    print('-----\nBest hyper-param strategy: {}'.format(population_model[best_individual_index].history_hp))
    print('final loss: {}'.format(population_model[best_individual_index].history_loss[-1]))
