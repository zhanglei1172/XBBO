'''
Reference: https://github.com/automl/DEHB
'''

from typing import List
import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.multi_fidelity.hyperband import HB
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import BasicConfigGenerator
from xbbo.search_algorithm.tpe_optimizer import TPE
from xbbo.utils.constants import MAXINT, Key
from .. import alg_register

class BOHB_CG(BasicConfigGenerator, TPE):
    def __init__(self, cs, budget, max_pop_size, rng, **kwargs) -> None:
        BasicConfigGenerator.__init__(self, cs, budget, max_pop_size, rng, **kwargs)
        TPE.__init__(self, space=cs, **kwargs)
        self.reset(max_pop_size)

alg_marker = 'bohb'

@alg_register.register(alg_marker)
class BOHB(HB):
    name = alg_marker
    def __init__(self,
                 space: DenseConfigurationSpace,
                 budget_bound=[9, 729],
                #  mutation_factor=0.5,
                #  crossover_prob=0.5,
                #  strategy='rand1_bin',
                 eta: int = 3,
                 seed: int = 42,
                 round_limit: int = 1,
                 bracket_limit=np.inf,
                 boundary_fix_type='random',
                 **kwargs):
        HB.__init__(self,
                    space,
                    budget_bound,
                    eta,
                    seed=seed,
                    round_limit=round_limit,
                    bracket_limit=bracket_limit,
                    boundary_fix_type=boundary_fix_type,
                    **kwargs)

        # Uniform2Gaussian.__init__(self,)
        # self._get_max_pop_sizes()
        # self._init_subpop(**kwargs)

    def _init_subpop(self, **kwargs):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.cg = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.cg[b] = BOHB_CG(
                self.space,
                seed=self.rng.randint(MAXINT),
                initial_design="random",
                init_budget=0,
                budget=b,
                max_pop_size=self._max_pop_size[b],
                rng=self.rng,
                **kwargs)
            # self.cg[b] = TPE(
            #     self.space,
            #     seed=self.rng.randint(MAXINT),
            #     initial_design="random",
            #     init_budget=0,
            #     **kwargs)
            # self.cg[b].population = [None] * self._max_pop_size[b]
            # self.cg[b].population_fitness = [np.inf] * self._max_pop_size[b]

    def _acquire_candidate(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
                # select a parent/target
        parent_id = self._get_next_idx_for_subpop(budget, bracket)

        if budget != bracket.budgets[0]:
            if bracket.is_new_rung():
                # TODO: check if generalizes to all budget spacings
                lower_budget, num_configs = bracket.get_lower_budget_promotions(
                    budget)
                self._get_promotion_candidate(lower_budget, budget,
                                              num_configs)
            # else: # 每一列中的第一行，随机生成config

        else:
            for b in reversed(self.budgets):
                if self.cg[b].kde_models:
                    break 
            trial = self.cg[b]._suggest()[0]

            self.cg[budget].population[parent_id] = trial.array
        # parent_id = self._get_next_idx_for_subpop(budget, bracket)

        target = self.cg[budget].population[parent_id]
        # target = self.fix_boundary(target)
        return target, parent_id



opt_class = BOHB
