import copy

import numpy as np
import matplotlib.pyplot as plt
import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.configspace.space import DenseConfiguration
from xbbo.search_algorithm.pbt_optimizer import PBT
from xbbo.core.constants import MAXINT

from examples.PopulationBasedTraining.toy_model import Model

class ToyPBT(PBT):
    def exploit_and_explore(self, population_model, losses):
        self.population_losses_his.append(losses)
        s_id = np.argsort(losses)
        top_num = max(int(self.fraction * len(s_id)), 1)
        top_ids = s_id[:top_num]
        bot_ids = s_id[-top_num:]
        # nobot_ids = s_id[:-self.fraction*len(s_id)]
        for bot_id in bot_ids:  # TODO Toy every point do explore
            # exploit
            top_id = np.random.choice(top_ids)
            checkpoint = population_model[top_id].save_checkpoint()
            population_model[bot_id].load_checkpoint(checkpoint)
            # self.population_hp_array[bot_id] = self.population_hp_array[top_id].copy() # TODO Toy
            # explore
            self.population_hp_array[bot_id] = np.clip(  # TODO Toy
                self.population_hp_array[top_id] +
                self.rng.normal(0, 0.1, size=self.dimension), 0, 1)

            new_config = DenseConfiguration.from_array(
                self.space, self.population_hp_array[bot_id])
            self.population_configs[bot_id] = new_config
            # self.population_hp[bot_id] = x_unwarped
            # population_model[bot_id].history_hp = copy.copy(
            # population_model[top_id].history_hp)
            population_model[bot_id].history_loss = copy.copy(
                population_model[top_id].history_loss)
            population_model[bot_id].update_hp(new_config.get_dictionary())



if __name__ == "__main__":
    epoch_num = 100
    rng = np.random.RandomState(42)

    config_space = Model.get_configuration_space(rng.randint(MAXINT))
    # define black box optimizer
    pbt = ToyPBT(space=config_space, pop_size=2, seed=rng.randint(MAXINT))
    population_model = [
        Model(seed=rng.randint(MAXINT)) for _ in range(pbt.pop_size)
    ]
    interval = 1
    pbt.init_model_hp(population_model)

    losses = pbt.optimize(population_model=population_model, epoch_num=epoch_num, interval=interval)


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
    # plt.savefig('./out/PBT_toy_.png')
    plt.show()

    print('-----\nBest hyper-param strategy: {}'.format(population_model[best_individual_index].history_hp))
    print('final loss: {}'.format(population_model[best_individual_index].history_loss[-1]))
