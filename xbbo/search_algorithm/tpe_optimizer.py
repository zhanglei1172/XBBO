import logging
import traceback
import numpy as np
import statsmodels.api as sm
import scipy.stats as sps
import ConfigSpace

from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace, deactivate_inactive_hyperparameters
from xbbo.core.trials import Trial, Trials
from xbbo.utils.constants import MAXINT
from . import alg_register
from xbbo.initial_design import ALL_avaliable_design

logger = logging.getLogger(__name__)


@alg_register.register('tpe')
class TPE(AbstractOptimizer):
    '''
    reference: https://github.com/thomas-young-2013/open-box/blob/master/openbox/core/tpe_advisor.py
    '''
    def __init__(
            self,
            space: DenseConfigurationSpace,
            seed: int = 42,
            #  bandwidth=1,
            gamma=0.15,
            initial_design: str = 'sobol',
            suggest_limit: int = np.inf,
            candidates_num=64,
            min_bandwidth=1e-3,
            bandwidth_factor=3,
            min_points_in_model=None,
            random_fraction=1 / 3,
            **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='round',
                                   encoding_ord='round',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)
        self.min_bandwidth = min_bandwidth
        self.bw_factor = bandwidth_factor

        self.dimension = self.space.get_dimensions()
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=suggest_limit, **kwargs)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )

        self.trials = Trials(dim=self.dimension)
        self.gamma = gamma
        self.candidates_num = candidates_num
        self.min_points_in_model = min_points_in_model

        hps = self.space.get_hyperparameters()

        if min_points_in_model is None:
            self.min_points_in_model = len(hps) + 1

        if self.min_points_in_model < len(hps) + 1:
            self.min_points_in_model = len(hps) + 1

        self.random_fraction = random_fraction
        self.kde_models = dict()

        self.kde_vartypes = ""
        self.vartypes = []

        for h in hps:
            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)

    def _suggest(self, n_suggestions=1):
        trial_list = []
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
                          array=config.get_array()))
        else:
            self._fit_kde_models()
            if len(self.kde_models.keys()
                   ) == 0 or self.rng.rand() < self.random_fraction:
                configs = self._sample_nonduplicate_config(n_suggestions)
                for config in configs:
                    trial_list.append(
                        Trial(configuration=config,
                              config_dict=config.get_dictionary(),
                              array=config.get_array()))
            else:
                for n in range(n_suggestions):
                    try:
                        best = np.inf
                        best_vector = None
                        l = self.kde_models['good'].pdf
                        g = self.kde_models['bad'].pdf

                        minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

                        kde_good = self.kde_models['good']
                        kde_bad = self.kde_models['bad']

                        for i in range(self.candidates_num):
                            idx = self.rng.randint(0, len(kde_good.data))
                            datum = kde_good.data[idx]
                            vector = []

                            for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                                bw = max(bw, self.min_bandwidth)
                                if t == 0:
                                    bw = self.bw_factor * bw
                                    try:
                                        vector.append(
                                            sps.truncnorm.rvs(-m / bw,
                                                            (1 - m) / bw,
                                                            loc=m,
                                                            scale=bw))
                                    except:
                                        logger.warning(
                                            "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s"
                                            % (datum, kde_good.bw, m))
                                        logger.warning("data in the KDE:\n%s" %
                                                    kde_good.data)
                                else:

                                    if self.rng.rand() < (1 - bw):
                                        vector.append(int(m))
                                    else:
                                        vector.append(self.rng.randint(t))
                            val = minimize_me(vector)

                            if not np.isfinite(val):
                                logger.warning(
                                    'sampled vector: %s has EI value %s' %
                                    (vector, val))
                                logger.warning("data in the KDEs:\n%s\n%s" %
                                            (kde_good.data, kde_bad.data))
                                logger.warning("bandwidth of the KDEs:\n%s\n%s" %
                                            (kde_good.bw, kde_bad.bw))
                                logger.warning("l(x) = %s" % (l(vector)))
                                logger.warning("g(x) = %s" % (g(vector)))

                                # right now, this happens because a KDE does not contain all values for a categorical parameter
                                # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                                # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
                                if np.isfinite(l(vector)):
                                    best_vector = vector
                                    break

                            if val < best:
                                best = val
                                best_vector = vector

                        if best_vector is None:
                            logger.debug(
                                "Sampling based optimization with %i samples failed -> using random configuration"
                                % self.candidates_num)
                            config = self._sample_nonduplicate_config()[0]
                        else:
                            logger.debug('best_vector: {}, {}, {}, {}'.format(
                                best_vector, best, l(best_vector), g(best_vector)))
                            for i, hp_value in enumerate(best_vector):
                                if isinstance(
                                        self.space.get_hyperparameter(
                                            self.space.get_hyperparameter_by_idx(
                                                i)), ConfigSpace.hyperparameters.
                                        CategoricalHyperparameter):
                                    best_vector[i] = int(np.rint(best_vector[i]))
                            config = DenseConfiguration.from_array(
                            self.space, np.asarray(best_vector))
                        try:
                            config = deactivate_inactive_hyperparameters(
                                        configuration_space=self.space,
                                        configuration=config.get_dictionary()
                                        )

                        except Exception as e:
                            logger.warning(("="*50 + "\n")*3 +\
                                    "Error converting configuration:\n%s"%config+\
                                    "\n here is a traceback:" +\
                                    traceback.format_exc())
                            raise(e)


                    except:
                        logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration"%(self.num_samples, traceback.format_exc()))
                        # config = self._sample_nonduplicate_config()[0]
                        config = self.space.sample_configuration()[0]
                    trial_list.append(
                            Trial(configuration=config,
                                config_dict=config.get_dictionary(),
                                array=config.get_array()))
        return trial_list

    def _sample_nonduplicate_config(self, num_configs=1):
        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.space.sample_configuration()[0]
            sample_cnt += 1
            if (not self.trials.is_contain(config)) and config not in configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning(
                    'Cannot sample non duplicate configuration after %d iterations.'
                    % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs

    def _fit_kde_models(self, ):
        train_configs = self.trials.get_array()
        n_good = max(self.min_points_in_model,
                     int(self.gamma * self.trials.trials_num) // 100)
        # n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
        n_bad = max(self.min_points_in_model,
                    int((1 - self.gamma) * self.trials.trials_num))

        # Refit KDE for the current budget
        idx = np.argsort(self.trials._his_observe_value)

        train_data_good = self.impute_conditional_data(
            train_configs[idx[:n_good]])
        train_data_bad = self.impute_conditional_data(
            train_configs[idx[n_good:n_good + n_bad]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        bw_estimation = 'normal_reference'
        # np.random.seed(self.rng.randint(MAXINT))
        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad,
                                                   var_type=self.kde_vartypes,
                                                   bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good,
                                                    var_type=self.kde_vartypes,
                                                    bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models = {'good': good_kde, 'bad': bad_kde}

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)

    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).ravel()

            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(
                    array[:, nan_idx])).ravel()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = self.rng.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = self.rng.rand()
                    else:
                        datum[nan_idx] = self.rng.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).ravel()
            return_array[i, :] = datum
        return return_array


opt_class = TPE