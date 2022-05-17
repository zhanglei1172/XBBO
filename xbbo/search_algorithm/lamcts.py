from typing import Optional, List, Tuple, cast

import numpy as np
from xbbo.alg_auxiliary.lamcts import MCTS
from xbbo.initial_design import ALL_avaliable_design
from xbbo.utils.constants import MAXINT
# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trial, Trials
from xbbo.alg_auxiliary.lamcts.latent_space import LatentConverterRNVP, LatentConverterVAE, LatentConverterPCA, LatentConverterCNN, LatentConverterIdentity
from . import alg_register


@alg_register.register('lamcts')
class LaMCTS(AbstractOptimizer):
    def __init__(
        self,
        space: DenseConfigurationSpace,
        seed: int = 42,
        initial_design: str = 'lh',
        init_budget: int = None,
        suggest_limit: int = np.inf,
        split_latent_model: str = 'identity',
        split_latent_dims: int = None,
        sample_latent_dims: int = None,
        sample_latent_model: str = 'identity',
        device: str = "cpu",
        leaf_size=20,
        kernel_type='rbf',
        gamma_type='auto',
        C_p: float = 10,
        solver="cmaes",
        split_metric='max',
        cmaes_sigma_mult=1.0,
        use_gpr=True,
        treeify_freq=1,
        init_within_leaf="mean",
        splitter_type="kmeans",
        normalize=True,
        split_use_predict=True,  # False->kmeans result; True->svm
        **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)
        # Uniform2Gaussian.__init__(self, )
        if self.space.get_conditions():
            raise NotImplementedError(
                "LaMCTS optimizer currently does not support conditional space!"
            )
        self.dimension = self.space.get_dimensions()
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space,
            self.rng,
            ta_run_limit=suggest_limit,
            init_budget=init_budget,
            **kwargs)
        self.init_budget = self.initial_design.init_budget
        self.initial_design_configs = self.initial_design.select_configurations(
        )
        self.bounds = self.space.get_bounds()
        if split_latent_model == 'identity':
            self.split_latent_converter = LatentConverterIdentity(
                self.bounds,
                dim=self.dimension,
                latent_dim=split_latent_dims,
                device=device,
                rng=self.rng,
                **kwargs)
        else:
            raise NotImplementedError
        # elif latent_model == 'pca':
        #     latent_converter = LatentConverterPCA(args, latent_dim, device=device, **kwargs)
        # elif latent_model == 'cnn':
        #     latent_converter = LatentConverterCNN(args, env_info, device=device)
        # elif latent_model == 'vae':
        #     latent_converter = LatentConverterVAE(args, env_info, device=device)
        # elif latent_model == 'realnvp':
        #     latent_converter = LatentConverterRNVP(args, env_info, device=device)
        # elif latent_model == 'identity':
        #     latent_converter = LatentConverterIdentity(args, env_info, device=device)
        if sample_latent_model == 'identity':
            self.sample_latent_converter = LatentConverterIdentity(
                self.bounds,
                dim=self.dimension,
                latent_dim=sample_latent_dims,
                device=device,
                rng=self.rng,
                **kwargs)
        else:
            raise NotImplementedError
        self.sample_latent_dims = self.sample_latent_converter.latent_dim
        self.split_latent_dims = self.split_latent_converter.latent_dim
        self.sample_latent_bounds = self.sample_latent_converter.bounds
        # self.bounds = self.space.get_bounds()
        # self.es = cma.CMAEvolutionStrategy([0.5] * self.dimension,
        #                                    0.1,
        #                                    inopts={
        #                                        'seed':
        #                                        self.rng.randint(MAXINT),
        #                                        'bounds': [0, 1]
        #                                    })
        # self.hp_num = len(configs)

        self.trials = Trials(space, dim=self.dimension)
        self.agent = MCTS(
            self.space,
            sample_latent_bounds=self.sample_latent_bounds,
            dims=self.dimension,
            split_latent_converter=self.split_latent_converter,
            sample_latent_converter=self.sample_latent_converter,
            split_latent_dims=self.split_latent_dims,
            sample_latent_dims=self.sample_latent_dims,
            solver=solver,
            split_metric=split_metric,
            cmaes_sigma_mult=cmaes_sigma_mult,
            use_gpr=use_gpr,
            treeify_freq=treeify_freq,
            init_within_leaf=init_within_leaf,
            splitter_type=splitter_type,
            C_p=C_p,
            leaf_size=leaf_size,
            kernel_type=kernel_type,
            gamma_type=gamma_type,
            normalize=normalize,
            rng=self.rng,
            split_use_predict=split_use_predict,
            **kwargs)
        # best_x, best_fx = agent.search(iterations = args.iterations, samples_per_iteration=args.samples_per_iteration, treeify_freq=args.treeify_freq)
        # assert func.counter == args.iterations
        # return best_x.reshape(args.horizon, env_info['action_dims']), agent

    def _suggest(self, n_suggestions=1):
        trial_list = []
        # currently only suggest one
        if (self.trials.trials_num) < self.init_budget:
            assert self.trials.trials_num % n_suggestions == 0
            configs = self.initial_design_configs[
                int(n_suggestions *
                    self.trials.trials_num):int(n_suggestions *
                                                (self.trials.trials_num + 1))]
            for config in configs:
                trial_list.append(
                    Trial(configuration=config,
                          config_dict=config.get_dictionary(),
                          array=config.get_array(),
                          _latent_sample=None,
                          _leaf=None))
        else:
            # if (self.trials.trials_num) < self.min_sample:
            #     while len(trial_list) < n_suggestions:  # remove history suggest
            #         config = self.cs.sample_configuration(size=1)[0]
            #         if not self.trials.is_contain(config):
            #             trial_list.append(
            #                 Trial(configuration=config,
            #                     config_dict=config.get_dictionary(),
            #                     array=config.get_array()))
            #     return trial_list
            leaf, latent_samples, samples = self.agent.suggest(n_suggestions)
            for n in range(n_suggestions):
                array = samples[n]
                config = DenseConfiguration.from_array(self.space, array)
                trial_list.append(
                    Trial(config,
                          config_dict=config.get_dictionary(),
                          array=array,
                          _latent_sample=latent_samples[n],
                          _leaf=leaf))

        return trial_list

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial)
            # self.listx.append(self.array_to_feature(trial.array))
            # self.listx.append(trial.array)
            # self.listy.append(trial.observe_value)
            self.agent.observe(trial._leaf, trial._latent_sample, trial.array,
                               -trial.observe_value)  # LaMCTS Maximize


opt_class = LaMCTS
