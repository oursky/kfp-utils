from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class OptimizationAlgorothm:
    name: str

    def to_algorithm_spec(self):
        output = {
            'algorithm': {
                'algorithmName': self.name,
            }
        }

        settings = self.get_settings()
        if settings:
            output['algorithmSettings'] = settings

        return output

    def get_settings(self) -> Dict:
        return {
            k: v
            for k, v in asdict(self).items()
            if k != 'name' and v is not None
        }


@dataclass
class GridSearch(OptimizationAlgorothm):
    name: str = 'grid'


@dataclass
class RandomSearch(OptimizationAlgorothm):
    name: str = 'random'
    random_state: Optional[int] = None


@dataclass
class BayesianOptimization(OptimizationAlgorothm):
    """
    For setting description, refer to
    https://www.kubeflow.org/docs/components/katib/experiment/#bayesian-optimization
    """

    name: str = 'bayesianoptimization'
    base_estimator: str = 'GP'
    n_initial_points: int = 10
    acq_func: str = 'gp_hedge'
    acq_optimizer: str = 'auto'
    random_state: Optional[int] = None


@dataclass
class Hyperband(OptimizationAlgorothm):
    name: str = 'hyperband'


@dataclass
class TPE(OptimizationAlgorothm):
    name: str = 'tpe'


@dataclass
class _CMAES:
    sigma: float
    random_state: Optional[int] = None
    restart_strategy: str = 'none'


@dataclass
class CMAES(OptimizationAlgorothm, _CMAES):
    name: str = 'cmaes'


@dataclass
class Sobol(OptimizationAlgorothm):
    name: str = 'sobol'
