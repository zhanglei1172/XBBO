import typing
from scipy.stats.qmc import Sobol

from xbbo.configspace.space import Configurations
from xbbo.initial_design.initialDesign import InitialDesign

class SobolDesign(InitialDesign):
    """ Sobol sequence design with a scrambled Sobol sequence.

    See https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html for further information

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    """

    def _select_configurations(self) -> typing.List[Configurations]:
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """

        sobol_gen = Sobol(d=self.dim, scramble=True, seed=self.rng.randint(low=0, high=10000000))
        sobol = sobol_gen.random(self.init_budget)
        return sobol

        # return self._transform_continuous_designs(design=sobol,
        #                                           origin='Sobol',
        #                                           cs=self.cs)
