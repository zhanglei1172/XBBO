import numpy as np
from time import time
import tqdm
from matplotlib import pyplot as plt

from xbbo.search_space import problem_register
from xbbo.search_algorithm import alg_register
from xbbo.configspace import build_space


class PBT:

    def __init__(self, cfg):
        # setup TestProblem
        self.cfg = cfg
        self.pop_size = cfg.OPTM.pop_size

        self.population_model = [problem_register[cfg.TEST_PROBLEM.name](cfg) for _ in range(self.pop_size)]
        self.api_config = self.population_model[0].get_api_config()  # 优化的hp
        self.config_spaces = build_space(self.api_config)

        # Setup optimizer
        opt_class = alg_register[cfg.OPTM.name]

        self.optimizer_instance = opt_class(self.config_spaces, self.pop_size, **dict(cfg.OPTM.kwargs))
        self.n_suggestions = cfg.OPTM.n_suggestions
        self.n_obj = cfg.OPTM.n_obj

        assert self.n_suggestions >= 1, "batch size must be at least 1"
        assert self.n_obj >= 1, "Must be at least one objective"

        self.epoch = cfg.OPTM.epoch
        self.interval = cfg.OPTM.interval

        # self.record = Record(self.cfg)

    def evaluate(self, population_model_history_hp):
        model = self.population_model[0]
        for step_num, params, acc in population_model_history_hp:
            model.update_hp(params)
            for i in range(step_num):
                model.step()

    def run(self):
        self.optimizer_instance.init_model_hp(self.population_model)
        finished = False
        with tqdm.tqdm(total=int(len(self.population_model[-1]) * self.epoch)) as pbar:
            while not finished:
                for i in range(self.pop_size):
                    self.population_model[i].step(int(self.interval * len(self.population_model[i])))
                    if self.population_model[i].step_num == int(len(self.population_model[i]) * self.epoch):
                        finished = True
                    # while True:
                    #     self.population_model[i].step()
                    #     if self.population_model[i].step_num % (self.interval * len(self.population_model[i])) == 0:
                    #         if self.population_model[i].step_num == len(self.population_model[i]) * self.epoch:
                    #             finished = True
                    #         # self.population_model[i].ready = True
                    #         break
                # asynchronous wait all active
                for i in range(self.pop_size):
                    self.population_model[i].evaluate()
                scores = [net.score for net in self.population_model]
                pbar.update(self.interval * len(self.population_model[-1]))
                if finished:
                    break
                self.optimizer_instance.exploit_and_explore(self.population_model, scores)
                # self.optimizer_instance.exploit_and_explore_toy(self.population_model, scores)
        return scores

    def show_res(self, scores):
        best_individual_index = np.argmax(scores)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for i in range(self.pop_size):
            desc_data = np.array(self.population_model[i].history_score)
            desc_data[:, 0] /= len(self.population_model[-1])
            ax1.plot(desc_data[:, 0], desc_data[:, 1], alpha=0.5)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("score")
        # for i in range(self.pop_size):
        #     desc_data = np.array([list(x[-1].values()) for x in self.population_model[i].trajectory_hp])
        #     # desc_data[:, 0] /= self.interval * len(self.population_model[-1])
        #     ax2.scatter(desc_data[:, 0], desc_data[:, 1], alpha=0.5)
        # ax2.set_xlabel("hp_1")
        # ax2.set_ylabel("hp_2")
        for i in range(self.pop_size):
            desc_data = np.array([[x[0], x[-1]['lr']] for x in self.population_model[i].history_hp])
            desc_data[:, 0] /= len(self.population_model[-1])
            desc_data = np.append(desc_data, [[self.epoch, desc_data[-1, 1]]], axis=0)
            ax2.plot(desc_data[:, 0], desc_data[:, 1], label='best individual' if i==best_individual_index else None)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("lr")
        plt.legend()
        plt.suptitle("PBT search (lr, momentum) in MNIST")
        plt.tight_layout()
        plt.savefig('./out/PBT_mnist.png')
        plt.show()

        print('-----\nBest hyper-param strategy: {}'.format(self.population_model[best_individual_index].history_hp))
        print('final score: {}'.format(self.population_model[best_individual_index].history_score[-1]))

    def show_toy_res(self, scores):
        best_individual_index = np.argmax(scores)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for i in range(self.pop_size):
            desc_data = np.array(self.population_model[i].history_score)
            desc_data[:, 0] /= len(self.population_model[-1])
            ax1.plot(desc_data[:, 0], desc_data[:, 1], alpha=0.5)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("score")
        for i in range(self.pop_size):
            desc_data = np.array(self.population_model[i].trajectory_theta)
            # desc_data[:, 0] /= self.interval * len(self.population_model[-1])
            ax2.scatter(desc_data[:, 0], desc_data[:, 1], s=2, alpha=0.5)
        # ax2.axis('equal')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel(r"$\theta_1$")
        ax2.set_ylabel(r"$\theta_2$")
        # for i in range(self.pop_size):
        #     desc_data = np.array([[x[0], x[-1]['lr']] for x in self.population_model[i].history_hp])
        #     desc_data[:, 0] /= len(self.population_model[-1])
        #     desc_data = np.append(desc_data, [[self.epoch, desc_data[-1, 1]]], axis=0)
        #     ax2.plot(desc_data[:, 0], desc_data[:, 1], label='best individual' if i==best_individual_index else None)
        # ax2.set_xlabel("epoch")
        # ax2.set_ylabel("lr")
        # plt.legend()
        plt.suptitle("PBT toy example")
        plt.tight_layout()
        plt.savefig('./out/PBT_toy.png')
        plt.show()

        print('-----\nBest hyper-param strategy: {}'.format(self.population_model[best_individual_index].history_hp))
        print('final score: {}'.format(self.population_model[best_individual_index].history_score[-1]))
