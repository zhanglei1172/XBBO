import copy
import torch
from torch.nn.parameter import Parameter
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt
import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from examples.PopulationBasedTraining.mnist_model import Model
from xbbo.configspace.space import DenseConfiguration


from xbbo.search_algorithm.pbt_optimizer import PBT

from xbbo.core.constants import MAXINT

class MnistPBT(PBT):
    def exploit_and_explore(self, population_model, losses):
        '''
        For specific problem, you should specify yourself implement
        '''
        self.population_losses_his.append(losses)
        s_id = np.argsort(losses)
        top_num = (min(max(1, int(self.fraction * len(s_id))), len(s_id)-1))
        top_ids = s_id[:top_num]
        bot_ids = s_id[-top_num:]
        # nobot_ids = s_id[:-self.fraction*len(s_id)]
        for bot_id in bot_ids:
            # exploit
            top_id = self.rng.choice(top_ids)
            checkpoint = population_model[top_id].save_checkpoint()
            population_model[bot_id].load_checkpoint(checkpoint)
            # Keep dataloader iter syncronize(when loss is nan or early stopping)
            population_model[bot_id].iter_train_loader = population_model[top_id].iter_train_loader
            
            self.population_hp_array[bot_id] = self.population_hp_array[
                top_id].copy()
            # explore
            self.population_hp_array[bot_id] = np.clip(
                self.population_hp_array[bot_id] +
                np.random.normal(0, 0.2, size=self.dimension), 0, 1)

            new_config = DenseConfiguration.from_array(
                self.space, self.population_hp_array[bot_id])
            self.population_configs[bot_id] = new_config
            # x_array = self.feature_to_array(self.population_hp_array[bot_id],
            #                                 self.sparse_dimension)
            # x_unwarped = DenseConfiguration.array_to_dict(self.space, x_array)
            # self.population_hp[bot_id] = x_unwarped
            population_model[bot_id].history_hp = copy.copy(
                population_model[top_id].history_hp)
            population_model[bot_id].history_loss = copy.copy(
                population_model[top_id].history_loss)
            population_model[bot_id].update_hp(new_config.get_dictionary())

        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch_num = 10
    rng = np.random.RandomState(42)
    
    config_space = Model.get_configuration_space(rng.randint(MAXINT))
    # define black box optimizer
    pbt = MnistPBT(space=config_space, pop_size=5, seed=rng.randint(MAXINT))

    population_model = [
        Model(seed=rng.randint(MAXINT), device=device) for _ in range(pbt.pop_size)
    ]
    interval = 1
    pbt.init_model_hp(population_model)

    losses = pbt.optimize(population_model=population_model, epoch_num=epoch_num, interval=interval)


    best_individual_index = np.argmin(losses)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(pbt.pop_size):
        desc_data = np.array(population_model[i].history_loss)
        desc_data[:, 0] /= len(population_model[-1])
        ax1.plot(desc_data[:, 0], -desc_data[:, 1], alpha=0.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("score")
    # for i in range(self.pop_size):
    #     desc_data = np.array([list(x[-1].values()) for x in self.population_model[i].trajectory_hp])
    #     # desc_data[:, 0] /= self.interval * len(self.population_model[-1])
    #     ax2.scatter(desc_data[:, 0], desc_data[:, 1], alpha=0.5)
    # ax2.set_xlabel("hp_1")
    # ax2.set_ylabel("hp_2")
    for i in range(pbt.pop_size):
        desc_data = np.array([[x[0], x[-1]['lr']] for x in population_model[i].history_hp])
        desc_data[:, 0] /= len(population_model[-1])
        desc_data = np.append(desc_data, [[epoch_num, desc_data[-1, 1]]], axis=0)
        ax2.plot(desc_data[:, 0], desc_data[:, 1], label='best individual' if i==best_individual_index else None)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("lr")
    plt.legend()
    plt.suptitle("PBT search (lr, momentum) in MNIST")
    plt.tight_layout()
    # plt.savefig('./a.png')
    plt.show()

    print('-----\nBest hyper-param strategy: {}'.format(population_model[best_individual_index].history_hp))
    print('final -score: {}'.format(population_model[best_individual_index].history_loss[-1]))